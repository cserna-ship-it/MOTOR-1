#!/usr/bin/env python3
"""
03_timing_reintento_motor1.py — Optimizador de timing de reintento (días 1-30, vectorizado)

Simula los 30 días candidatos (1–30) en una sola inferencia batch:
177k pledges × 30 días = 5.3M filas, ~60s de ejecución.

Reglas de negocio aplicadas:
- Regla 21 días dinámica: para pledges con último intento exitoso,
  dia_min = max(1, 21 - dias_desde_ultimo). Para fallos, dia_min = 1.
- Tolerancia logarítmica: umbral = 0.016 × log2(d_extra+1). Si un día
  anterior al óptimo queda dentro del umbral, se recomienda ese día más cercano.

VENTANA DE HISTORIA (v9)
------------------------
El último intento de cada pledge se busca dentro de una ventana móvil de N meses
que termina en la fecha de corte: `(CUTOFF - N meses, CUTOFF]`, con N=12 por
defecto. Reemplaza al filtro por año calendario, que daba ventanas de ancho
variable (y vacías al correr en enero). Ver get_window_months().

USO
---
    python3 scripts/03_train/03_timing_reintento_motor1.py \\
        --cutoff-date 2025-12-31 --run-date 2026-01-03

    # Ventana más corta (solo comportamiento reciente)
    python3 scripts/03_train/03_timing_reintento_motor1.py \\
        --cutoff-date 2026-02-05 --run-date 2026-02-05 --window-months 6

Parámetros:
    --cutoff-date     frontera de información: hasta cuándo se conocen los intentos
    --run-date        "hoy": desde aquí se proyectan las fechas recomendadas
    --window-months   ancho de la ventana de historia (default 12)
    --features-file   override del parquet de features (p.ej. el YTD mensual)

Genera:
    results/motor1/timing_reintento_motor1.parquet
    results/motor1/timing_reintento_motor1.csv
"""
from pathlib import Path
import sys
import math
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import get_cutoff_date
import argparse

ROOT        = Path(__file__).resolve().parents[2]
ALLOC_FEAT  = ROOT / "data/features/allocation_features.parquet"
MODEL_PATH  = ROOT / "results/motor1/motor1_catboost.cbm"
CALIB_PATH  = ROOT / "results/motor1/motor1_calibrador.pkl"
OUT         = ROOT / "results/motor1"

# Días a simular (curva completa)
TODOS_LOS_DIAS = list(range(1, 31))

# Regla de negocio: mínimo 21 días entre el último intento exitoso y el próximo cobro
DIAS_MIN_EXITOSO = 21

# Tolerancia logarítmica: umbral(d_extra) = LOG_FACTOR × log2(d_extra + 1)
# Máximo ~10.8pp a d_extra=29
LOG_FACTOR = 0.016

FEAT_NUM = ["pre_failure_streak", "days_since_last_attempt",
            "success_count_cum", "failure_count_cum", "payment_amount",
            "month", "day_of_month", "day_of_week", "contact_age_days",
            "alloc_n_total", "alloc_n_downgrades_total", "alloc_n_method_changes_total",
            "alloc_n_day_changes_total", "alloc_n_frequency_changes_total",
            "alloc_n_errors_total", "alloc_days_since_last",
            "alloc_recent90_downgrade", "alloc_recent90_method_change",
            "alloc_recent90_day_change", "alloc_recent90_frequency_change",
            "alloc_recent90_error"]
FEAT_CAT = ["payment_gateway", "payment_method", "bank_canonical",
            "state_norm", "gender_norm", "age_band", "ses_level",
            "alloc_last_type", "alloc_last_status"]
FEATURES = FEAT_NUM + FEAT_CAT


def get_run_date() -> pd.Timestamp:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-date", type=str, default=None)
    args, _ = parser.parse_known_args()
    raw = args.run_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    run_date = pd.Timestamp(raw)
    print(f"  [config] RUN_DATE    = {run_date.date()}")
    return run_date


def get_window_months() -> int:
    """
    Ancho de la ventana de historia, en meses, sobre la que se busca el ÚLTIMO
    intento de cada pledge.

    POR QUÉ UNA VENTANA MÓVIL Y NO EL AÑO CALENDARIO (v9)
    -----------------------------------------------------
    Antes este script filtraba `attempt_ts.dt.year == CUTOFF.year`, lo que hacía
    que el tamaño real de la ventana dependiera del mes del corte:

        corte 2025-12-31 → 12 meses de historia (año completo)
        corte 2026-02-05 → 5 semanas de historia (solo enero + inicio de feb)
        corte 2026-01-03 → CERO intentos (el año apenas empezó) → lista vacía

    Con la ventana móvil el ancho es constante: `(CUTOFF - N meses, CUTOFF]`.
    Con N=12 y un corte a fin de año el resultado es prácticamente idéntico al
    filtro por año calendario, así que el comportamiento histórico se conserva.

    El límite superior es el corte: el corte es la frontera de información
    ("esto es lo que sé al día de hoy"). Si quieres incluir intentos posteriores,
    mueve el corte, no la ventana.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--window-months", type=int, default=12)
    args, _ = parser.parse_known_args()
    n = args.window_months
    if n < 1:
        raise SystemExit(f"[ERROR] --window-months debe ser >= 1 (recibido: {n})")
    print(f"  [config] WINDOW      = {n} meses de historia")
    return n


def get_features_file() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--features-file", type=str, default=None)
    args, _ = parser.parse_known_args()
    if args.features_file:
        p = Path(args.features_file)
        print(f"  [config] FEATURES    = {p}  (override)")
        return p
    default = ROOT / "data/features/motor1_features.parquet"
    print(f"  [config] FEATURES    = {default}  (default)")
    return default


def tolerancia_log(d_extra: int) -> float:
    return LOG_FACTOR * math.log2(d_extra + 1)


def candidatos_exitoso(dias_desde_ultimo: int) -> list:
    dia_min = max(1, DIAS_MIN_EXITOSO - dias_desde_ultimo)
    candidatos = [d for d in TODOS_LOS_DIAS if d >= dia_min]
    return candidatos if candidatos else [TODOS_LOS_DIAS[-1]]


def main():
    t_start = time.time()
    CUTOFF      = get_cutoff_date()
    RUN_DATE    = get_run_date()
    WINDOW_MONTHS = get_window_months()
    FEATURES_FILE = get_features_file()

    # Ventana móvil de historia: (VENTANA_INI, CUTOFF]. Abierta a la izquierda para
    # que dos ventanas consecutivas no compartan el día de la frontera.
    VENTANA_INI = CUTOFF - pd.DateOffset(months=WINDOW_MONTHS)
    print(f"  Ventana de intentos: ({VENTANA_INI.date()}, {CUTOFF.date()}]")

    print("[1/6] Cargando modelo y calibrador...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    with open(CALIB_PATH, "rb") as f:
        calibrador = pickle.load(f)
    print("  OK")

    print("[2/6] Cargando features del último intento por pledge...")
    cols_needed = ["pledge_id", "payment_id", "attempt_ts", "is_success",
                   "pre_failure_streak", "failure_count_cum", "success_count_cum",
                   "payment_amount", "contact_age_days",
                   "payment_gateway", "payment_method", "bank_canonical",
                   "state_norm", "gender_norm", "age_band", "ses_level"]

    def en_ventana(c):
        """Filtra un chunk a la ventana móvil (VENTANA_INI, CUTOFF]."""
        return c[(c["attempt_ts"] > VENTANA_INI) & (c["attempt_ts"] <= CUTOFF)]

    is_ytd_file = "ytd" in FEATURES_FILE.name.lower()
    if is_ytd_file:
        # El archivo YTD ya viene recortado al año en curso, pero se aplica la
        # misma ventana para que el resultado no dependa de cuál de los dos
        # archivos de features se use.
        df = en_ventana(pd.read_parquet(FEATURES_FILE, columns=cols_needed))
    else:
        pf = pq.ParquetFile(FEATURES_FILE)
        chunks = []
        for batch in pf.iter_batches(columns=cols_needed, batch_size=500_000):
            c = en_ventana(batch.to_pandas())
            if len(c):
                chunks.append(c)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=cols_needed)
    print(f"  Intentos cargados: {len(df):,}")

    if len(df) == 0:
        raise SystemExit(
            f"\n[ERROR] No hay intentos en la ventana ({VENTANA_INI.date()}, {CUTOFF.date()}].\n"
            f"        Revisa que el corte esté dentro del período con datos, o amplía la "
            f"ventana con --window-months N.\n")
    print(f"  Rango real de los intentos en ventana: "
          f"{df['attempt_ts'].min():%Y-%m-%d} → {df['attempt_ts'].max():%Y-%m-%d}")

    alloc = pq.read_table(ALLOC_FEAT).to_pandas()
    df = df.merge(alloc, on="payment_id", how="left")

    alloc_num = ["alloc_n_total","alloc_n_downgrades_total","alloc_n_method_changes_total",
                 "alloc_n_day_changes_total","alloc_n_frequency_changes_total","alloc_n_errors_total",
                 "alloc_days_since_last","alloc_recent90_downgrade","alloc_recent90_method_change",
                 "alloc_recent90_day_change","alloc_recent90_frequency_change","alloc_recent90_error"]
    alloc_cat = ["alloc_last_type","alloc_last_status"]
    for c in alloc_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1)
    for c in alloc_cat:
        if c in df.columns:
            df[c] = df[c].astype(object).fillna("UNKNOWN").astype(str)
    for c in ["payment_gateway","payment_method","bank_canonical",
              "state_norm","gender_norm","age_band","ses_level"]:
        if c in df.columns:
            df[c] = df[c].astype(object).fillna("UNKNOWN").astype(str)

    ultimo_intento = (df.sort_values(["pledge_id", "attempt_ts"])
                      .drop_duplicates("pledge_id", keep="last")
                      .reset_index(drop=True))
    n_f = (ultimo_intento["is_success"] == 0).sum()
    n_e = (ultimo_intento["is_success"] == 1).sum()
    print(f"  Pledges activos: {len(ultimo_intento):,}  (fallido: {n_f:,} | exitoso: {n_e:,})")

    print("[3/6] Construyendo DataFrame vectorizado (pledge × día)...")
    # Expandir cada pledge a 30 filas (una por día candidato)
    # Vectorizado: sin loops Python por pledge, solo operaciones pandas/numpy
    base = ultimo_intento.copy()
    base["dias_transcurridos"] = base["attempt_ts"].apply(
        lambda ts: max(0, (RUN_DATE - pd.Timestamp(ts)).days)
    )
    base["ultimo_exitoso"] = (base["is_success"] == 1)

    # Replicar base × 30 días
    base_rep = base.loc[base.index.repeat(len(TODOS_LOS_DIAS))].reset_index(drop=True)
    dias_rep  = np.tile(TODOS_LOS_DIAS, len(base))
    base_rep["dia_candidato"] = dias_rep

    # Fechas de cobro simuladas
    base_rep["fecha_cobro"] = pd.to_datetime(
        base_rep["dia_candidato"].apply(lambda d: RUN_DATE + pd.Timedelta(days=int(d)))
    )
    base_rep["total_dias"] = base_rep["dias_transcurridos"] + base_rep["dia_candidato"]

    # Features temporales
    base_rep["days_since_last_attempt"] = base_rep["total_dias"].astype(float)
    base_rep["day_of_month"]            = base_rep["fecha_cobro"].dt.day
    base_rep["day_of_week"]             = base_rep["fecha_cobro"].dt.dayofweek
    base_rep["month"]                   = base_rep["fecha_cobro"].dt.month
    base_rep["contact_age_days"]        = (
        base_rep["contact_age_days"].fillna(0).astype(float) + base_rep["total_dias"]
    )

    # Contadores según tipo de último intento
    exitoso_mask = base_rep["ultimo_exitoso"]
    base_rep["pre_failure_streak"] = np.where(
        exitoso_mask, 0.0,
        base_rep["pre_failure_streak"].astype(float) + 1
    )
    base_rep["failure_count_cum"] = np.where(
        exitoso_mask,
        base_rep["failure_count_cum"].astype(float),
        base_rep["failure_count_cum"].astype(float) + 1
    )
    base_rep["success_count_cum"] = np.where(
        exitoso_mask,
        base_rep["success_count_cum"].astype(float) + 1,
        base_rep["success_count_cum"].astype(float)
    )

    print(f"  DataFrame simulado: {len(base_rep):,} filas ({len(ultimo_intento):,} pledges × 30 días)")

    print("[4/6] Inferencia batch (predict_proba)...")
    t_inf = time.time()
    X = base_rep[FEATURES].copy()
    probs_crudas = model.predict_proba(X)[:, 1]
    probs_cal    = calibrador.transform(probs_crudas)
    base_rep["prob_cal"] = probs_cal
    print(f"  Inferencia completada en {time.time()-t_inf:.1f}s")

    print("[5/6] Pivotando resultados y calculando día óptimo...")
    # Pivot: una fila por pledge, columnas prob_d1...prob_d30
    pivot = base_rep.pivot_table(
        index="pledge_id", columns="dia_candidato", values="prob_cal", aggfunc="first"
    )
    pivot.columns = [f"prob_d{int(c)}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Metadatos del último intento
    meta = base[["pledge_id","attempt_ts","is_success","dias_transcurridos","ultimo_exitoso"]].copy()
    meta = meta.rename(columns={"attempt_ts": "ultimo_intento_ts"})
    pivot = pivot.merge(meta, on="pledge_id", how="left")

    prob_cols = [f"prob_d{d}" for d in TODOS_LOS_DIAS]

    def elegir_optimo(row):
        ultimo_exitoso = bool(row["ultimo_exitoso"])
        dias_t         = int(row["dias_transcurridos"])

        dias_validos = (candidatos_exitoso(dias_t) if ultimo_exitoso
                        else TODOS_LOS_DIAS)

        probs = {d: row[f"prob_d{d}"] for d in dias_validos}
        prob_max = max(probs.values())
        dia_max  = max(probs, key=probs.get)

        for d in dias_validos:
            d_extra = dia_max - d
            umbral  = tolerancia_log(d_extra) if d_extra > 0 else 0.0
            if probs[d] >= prob_max - umbral:
                return pd.Series({
                    "dias_esperar_desde_hoy": d,
                    "prob_optimo":            round(probs[d], 4),
                    "prob_max_absoluto":      round(prob_max, 4),
                })
        return pd.Series({
            "dias_esperar_desde_hoy": dia_max,
            "prob_optimo":            round(prob_max, 4),
            "prob_max_absoluto":      round(prob_max, 4),
        })

    optimos = pivot.apply(elegir_optimo, axis=1)
    pivot = pd.concat([pivot, optimos], axis=1)

    pivot["tipo_ultimo_intento"] = pivot["ultimo_exitoso"].map({True: "exitoso", False: "fallo"})
    pivot["fecha_recomendada"]   = pivot["dias_esperar_desde_hoy"].apply(
        lambda d: (RUN_DATE + pd.Timedelta(days=int(d))).date()
    )
    pivot["dias_desde_ultimo_hoy"] = pivot["dias_transcurridos"]

    col_order = (["pledge_id","ultimo_intento_ts","tipo_ultimo_intento",
                  "dias_desde_ultimo_hoy","dias_esperar_desde_hoy",
                  "fecha_recomendada","prob_optimo","prob_max_absoluto"]
                 + prob_cols)
    out_df = (pivot[col_order]
              .sort_values("prob_optimo", ascending=False)
              .reset_index(drop=True))

    print("[6/6] Guardando resultados...")
    out_df.to_parquet(OUT / "timing_reintento_motor1.parquet", index=False)
    out_df.to_csv(OUT / "timing_reintento_motor1.csv", index=False)

    t_total = time.time() - t_start
    print(f"\n  Total pledges: {len(out_df):,}")
    print(f"  Distribución días recomendados:")
    print(out_df["dias_esperar_desde_hoy"].value_counts().sort_index().to_string())
    adelantados = (out_df["prob_max_absoluto"] > out_df["prob_optimo"]).sum()
    print(f"  Pledges adelantados por tolerancia log: {adelantados:,} ({adelantados/len(out_df)*100:.1f}%)")
    print(f"  Tiempo total: {t_total:.1f}s")
    print(f"  [OK] {OUT / 'timing_reintento_motor1.parquet'}")
    print(f"  [OK] {OUT / 'timing_reintento_motor1.csv'}")


if __name__ == "__main__":
    main()
