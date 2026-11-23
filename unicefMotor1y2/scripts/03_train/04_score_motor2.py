#!/usr/bin/env python3
"""
04_score_motor2.py — Puntuación mensual de Motor 2 (NO reentrena)

PARA QUÉ SIRVE
--------------
Responde la pregunta operativa del mes: **¿qué donantes van a cancelar en los
próximos 30 días?**, usando el modelo ya entrenado.

Entrenar y puntuar son dos cosas distintas:

  ENTRENAR (02_train_motor2.py)  → el modelo aprende de meses cuyo desenlace ya
                                   conocemos. Necesita el label. Se hace una vez
                                   cada 6-12 meses, cuando hay un período cerrado
                                   contra el cual medirse.

  PUNTUAR (este script)          → se aplica el modelo guardado al snapshot del
                                   mes en curso, cuyo desenlace AÚN NO ocurrió.
                                   No necesita label. Se hace cada mes, en segundos.

Ventajas de separarlos:
  - Las métricas del modelo siguen siendo las medidas contra un año completo
    (ver resultados_motor2.json), no contra un mes sin desenlace.
  - El modelo NO se mueve mes a mes: un P2 de 0.75 en febrero significa lo mismo
    que un 0.75 en enero, porque salen del mismo árbol y del mismo calibrador.
    Reentrenar cada mes rompería esa comparabilidad.

CICLO MENSUAL
-------------
1. Cargar los pagos del mes que cerró y regenerar el staging
   (00_unify_raw_data.py + 01_create_staging.py con el archivo nuevo en raw/).
2. Generar SOLO el snapshot del mes en curso:

       python3 scripts/02_features_motor2/01_motor2_dataset.py \\
           --cutoff-date 2026-02-05 --months 1 \\
           --out data/features/motor2_snapshot_2026_02_01.parquet

3. Puntuar:

       python3 scripts/03_train/04_score_motor2.py --score-date 2026-02-05

`--score-date` es "hoy". El snapshot que se puntúa es el PRIMER DÍA de ese mes
(2026-02-05 → snapshot 2026-02-01), y sus features usan únicamente intentos
anteriores al 1 de febrero. Por eso conviene correrlo en los primeros días del
mes: el snapshot ya tiene toda la historia del mes anterior.

ENTRADA / SALIDA
----------------
Lee:    results/motor2/motor2_catboost.cbm      → modelo entrenado
        results/motor2/motor2_calibrador.pkl    → calibrador isotónico
        results/motor2/resultados_motor2.json   → (opcional) corte del modelo, para trazar
        data/features/motor2_snapshot_<AAAA_MM_01>.parquet
Genera: results/motor2/scoring_motor2_<AAAA_MM_DD>.parquet
        results/motor2/scoring_motor2_<AAAA_MM_DD>.csv

NO se emite el label: el mes que se puntúa no tiene desenlace todavía. Cualquier
"churn" que apareciera ahí serían solo las cancelaciones ya registradas en el CRM,
lo que daría una lectura falsamente optimista del acierto del modelo.
"""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results/motor2/motor2_catboost.cbm"
CALIB_PATH = ROOT / "results/motor2/motor2_calibrador.pkl"
METRICS_PATH = ROOT / "results/motor2/resultados_motor2.json"
OUT = ROOT / "results/motor2"

# Columnas de contexto que acompañan al puntaje. No entran al modelo: van para que
# el equipo de retención entienda POR QUÉ un pledge está arriba en la lista.
COLS_CONTEXTO = ["streak_actual", "tenure_days", "days_since_last_attempt",
                 "n_fallos_30d", "n_intentos_30d", "success_rate_hist",
                 "last_payment_amount", "last_payment_method", "last_bank_name"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Puntúa el snapshot del mes con el modelo Motor 2 ya entrenado.")
    p.add_argument("--score-date", type=str, default=None,
                   help="Fecha de ejecución 'hoy' (YYYY-MM-DD). Se puntúa el snapshot "
                        "del primer día de ese mes. Default: fecha del sistema.")
    p.add_argument("--dataset", type=str, default=None,
                   help="Parquet con el snapshot a puntuar. Default: "
                        "data/features/motor2_snapshot_<AAAA_MM_01>.parquet")
    return p.parse_args()


def preparar_features(df):
    """
    Tratamiento de nulos IDÉNTICO al de 02_train_motor2.py.

    Si esto se desincroniza del entrenamiento, el modelo recibe valores que nunca
    vio y el puntaje se degrada sin ningún error visible. Cualquier cambio acá
    tiene que replicarse allá (y viceversa).
    """
    for c in ["last_payment_method", "last_bank_name", "last_gateway"]:
        df[c] = df[c].astype(str).fillna("Unknown")
    df["last_payment_amount"] = df["last_payment_amount"].fillna(0).astype("float32")
    df["days_since_last_attempt"] = df["days_since_last_attempt"].fillna(999).astype("float32")
    df["last_attempt_success"] = df["last_attempt_success"].fillna(-1).astype("int8")
    return df


def main():
    args = parse_args()
    SCORE_DATE = pd.Timestamp(args.score_date) if args.score_date else pd.Timestamp.today().normalize()
    SNAPSHOT = SCORE_DATE.replace(day=1)
    SLUG_SNAP = SNAPSHOT.strftime("%Y_%m_%d")
    SLUG_RUN = SCORE_DATE.strftime("%Y_%m_%d")
    dataset = Path(args.dataset) if args.dataset else \
        ROOT / f"data/features/motor2_snapshot_{SLUG_SNAP}.parquet"

    print(f"  [scoring] fecha de ejecución = {SCORE_DATE.date()}  →  "
          f"snapshot a puntuar = {SNAPSHOT.date()}")

    # === Modelo y calibrador ===
    for path, que in [(MODEL_PATH, "modelo"), (CALIB_PATH, "calibrador")]:
        if not path.exists():
            raise SystemExit(
                f"\n[ERROR] Falta el {que} de Motor 2: {path}\n"
                f"        Entrena primero: python3 scripts/03_train/02_train_motor2.py\n")
    print("[1/4] Cargando modelo entrenado...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    with open(CALIB_PATH, "rb") as f:
        calibrador = pickle.load(f)

    # Trazabilidad: con qué corte se entrenó el modelo que estamos usando.
    corte_modelo = None
    if METRICS_PATH.exists():
        try:
            corte_modelo = json.load(open(METRICS_PATH)).get("corte", {})
        except (json.JSONDecodeError, OSError):
            pass
    if corte_modelo:
        print(f"  Modelo entrenado con corte {corte_modelo.get('cutoff_date')} "
              f"(train {corte_modelo.get('train_years')}, test {corte_modelo.get('test_year')})")
    # El orden de features lo dicta el modelo, no una lista escrita a mano: así no
    # hay forma de puntuar con las columnas desalineadas.
    features = list(model.feature_names_)
    print(f"  Features del modelo: {len(features)}")

    # === Snapshot a puntuar ===
    if not dataset.exists():
        raise SystemExit(
            f"\n[ERROR] No existe el snapshot a puntuar: {dataset}\n"
            f"        Genéralo con:\n"
            f"          python3 scripts/02_features_motor2/01_motor2_dataset.py \\\n"
            f"              --cutoff-date {SCORE_DATE.date()} --months 1 \\\n"
            f"              --out {dataset.relative_to(ROOT)}\n")
    print(f"[2/4] Cargando snapshot: {dataset.name}")
    df = pd.read_parquet(dataset)
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"])
    print(f"  Filas: {len(df):,}  |  snapshots en el archivo: "
          f"{sorted(s.strftime('%Y-%m-%d') for s in df['snapshot_ts'].unique())}")

    df = df[df["snapshot_ts"] == SNAPSHOT].copy()
    if len(df) == 0:
        raise SystemExit(
            f"\n[ERROR] El archivo no contiene el snapshot {SNAPSHOT.date()}.\n"
            f"        Regenéralo con --cutoff-date {SCORE_DATE.date()} --months 1.\n")
    print(f"  Pledges activos a puntuar: {len(df):,}")

    faltan = [c for c in features if c not in df.columns]
    if faltan:
        raise SystemExit(
            f"\n[ERROR] Al snapshot le faltan features que el modelo espera: {faltan}\n"
            f"        Regenera el snapshot con la versión actual de "
            f"01_motor2_dataset.py.\n")

    # === Puntuación ===
    print("[3/4] Puntuando...")
    df = preparar_features(df)
    prob_cruda = model.predict_proba(df[features])[:, 1]
    prob = calibrador.transform(prob_cruda)   # calibrada ← la que se usa

    out = df[["pledge_id", "snapshot_ts"]].copy()
    out["prob_churn"] = prob.astype("float32")
    out["prob_churn_cruda"] = prob_cruda.astype("float32")
    for c in COLS_CONTEXTO:
        if c in df.columns:
            out[c] = df[c].values
    out = out.sort_values("prob_churn", ascending=False).reset_index(drop=True)

    print("[4/4] Guardando...")
    pq_path = OUT / f"scoring_motor2_{SLUG_RUN}.parquet"
    csv_path = OUT / f"scoring_motor2_{SLUG_RUN}.csv"
    out.to_parquet(pq_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"\n  Snapshot puntuado: {SNAPSHOT.date()} "
          f"(riesgo de cancelación entre {SNAPSHOT.date()} y "
          f"{(SNAPSHOT + pd.Timedelta(days=30)).date()})")
    print(f"  Pledges: {len(out):,}   |   P2 medio: {out['prob_churn'].mean():.4f}")
    print(f"  Tamaño de la lista de llamadas según umbral:")
    for u in (0.8, 0.6, 0.4, 0.2):
        n = int((out["prob_churn"] > u).sum())
        print(f"    P2 > {u:.1f}:  {n:>7,} pledges  ({100 * n / len(out):.2f}%)")
    print(f"\n  [OK] {csv_path}")
    print(f"  [OK] {pq_path}")


if __name__ == "__main__":
    main()
