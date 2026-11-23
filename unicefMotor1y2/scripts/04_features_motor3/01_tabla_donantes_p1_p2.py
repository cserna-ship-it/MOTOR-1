#!/usr/bin/env python3
"""
01_tabla_donantes_p1_p2.py — Ensamble final del entregable

PROPÓSITO
---------
Generar la TABLA MAESTRA DEL ENTREGABLE: una fila por donante (pledge)
activo en 2025, con sus probabilidades P1 (éxito de cobro) y P2 (churn 30d)
junto con todos los atributos del donante necesarios para análisis de
política / priorización.

ESTRUCTURA DE LA TABLA FINAL (36 cols, ~177.859 filas)
-----------------------------------------------------
Identificación:           pledge_id, contact_id
Metadata del contrato:    pledge_status, pledge_start_date, pledge_end_date,
                          tenure_days_at_2025_12_31, tipo_donante
P1 y P2:                  p1_promedio, p1_ultimo_intento,
                          p2_ultimo, p2_max, p2_dic_2025
Comportamiento histórico: n_intentos_2025, n_exitos_2025, tasa_exito_2025,
                          pre_failure_streak, failure_count_cum,
                          success_count_cum
Atributos del intento:    ultimo_attempt_ts, payment_amount, payment_gateway,
                          payment_method, payment_submethod, bank_canonical
Demografía:               state_norm, gender_norm, age_band, ses_level,
                          contact_age_days
Allocation:               alloc_n_total, alloc_n_downgrades_total,
                          alloc_recent90_downgrade, alloc_last_type,
                          alloc_days_since_last
Snapshots de referencia:  primer_attempt_ts, ultimo_snapshot_ts

REGLAS DE AGREGACIÓN
--------------------
P1 (probabilidad de éxito de cobro), sobre los intentos del año de agregación
(el año del corte, o el último año con datos si el del corte está vacío):
    p1_promedio        = MEAN(prob_success) sobre los intentos del pledge
    p1_ultimo_intento  = prob_success del último intento del pledge

P2 (probabilidad de churn 30d):
    p2_ultimo          = prob_churn del snapshot MÁS RECIENTE del histórico
    p2_max             = MAX(prob_churn) del histórico (peor momento)
    p2_<mmm>_<aaaa>    = el snapshot puntual accionable. Con `--p2-scoring` sale
                         del scoring del MES EN CURSO (a quién llamar ahora); sin
                         el parámetro, del snapshot del mes de corte dentro del
                         histórico de test (mes ya cerrado, sirve para validar el
                         modelo contra cancelaciones reales).
                         El nombre lleva el mes en inglés: p2_dec_2025, p2_jan_2026.

PARÁMETROS
----------
    --cutoff-date    fecha de corte: nombra el archivo y la columna de tenure, y
                     define el año de agregación de P1
    --p2-scoring     (opcional) parquet de scoring mensual de Motor 2

Próximo cobro recomendado (timing óptimo, salida de 03_timing_reintento_motor1):
    dias_proximo_cobro        = días a esperar desde hoy para el siguiente cobro
    fecha_proximo_cobro       = fecha concreta recomendada (hoy + dias_proximo_cobro)
    prob_exito_proximo_cobro  = prob. de éxito estimada en ese día óptimo

Atributos del último intento:
    Se toma el último intento del pledge en 2025 como "snapshot de
    atributos" — banco, gateway, monto, etc. usados al final del año.

Tipo de donante (segmentación operacional):
    "Prospecting" si tenure_days_at_2025_12_31 < 365 (donantes < 1 año)
    "House" si tenure_days_at_2025_12_31 >= 365 (donantes consolidados)
    

CASOS DE USO 
----------------------------
1. Validación manual: ¿los donantes con P2 > 80% en dic 2025 efectivamente
   cancelaron en Q1 2026?
2. Priorización: ¿qué donantes tienen P1 alto + P2 alto? → focos de
   retención urgente (cobrables pero a punto de churnear).
3. Cruce con datos propios: pledge_id permite hacer joins con sistemas
   operativos del cliente.
4. Auditoría: cada métrica del modelo es trazable hasta el donante.

NOTA v8 — TENURE Y MÉTODO DE PAGO
---------------------------------
- `tenure_days_at_2025_12_31` se ancla ahora a la PRIMERA evidencia real de
  actividad del donante: min(pledge_start_date, primer intento observado en
  toda la historia). Antes usaba solo `npsp__Start_Date__c`, que en ~1.45%
  de los pledges es una fecha FUTURA (ciclo/renovación) y producía tenure
  negativo pese a que el donante llevaba todo el año pagando. Ver detalle
  abajo en el cálculo.
- Se incluye `payment_method` (medio de pago) en la tabla; en v3 estaba
  marcado como deprecado y se excluía.

ENTRADA / SALIDA
----------------
Lee:    data/features/motor1_features.parquet
        data/features/allocation_features.parquet
        results/motor1/test_predictions_motor1.parquet
        results/motor2/test_predictions_motor2.parquet
        results/motor2/scoring_motor2_<AAAA_MM_DD>.parquet (opcional; --p2-scoring)
        results/motor1/timing_reintento_motor1.parquet (opcional; próximo cobro)
        data/unified/pledge/pledge_unified.parquet
Genera: data/entregable/Tabla_Donantes_P1_P2_<CUTOFF>.parquet
        data/entregable/Tabla_Donantes_P1_P2_<CUTOFF>.csv (mismo, en CSV
        para inspección en Excel)
"""
from pathlib import Path
import argparse
import sys
import gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/ al path
from config import get_cutoff_date

MOTOR1_FEATURES = ROOT / "data/features/motor1_features.parquet"
ALLOC_FEAT = ROOT / "data/features/allocation_features.parquet"
PRED_M1 = ROOT / "results/motor1/test_predictions_motor1.parquet"
PRED_M2 = ROOT / "results/motor2/test_predictions_motor2.parquet"
TIMING  = ROOT / "results/motor1/timing_reintento_motor1.parquet"
PLEDGE = ROOT / "data/unified/pledge/pledge_unified.parquet"

OUT_DIR = ROOT / "data/entregable"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_extra_args():
    """
    --p2-scoring RUTA   Parquet de scoring mensual de Motor 2
                        (results/motor2/scoring_motor2_<AAAA_MM_DD>.parquet).

        Sin este parámetro, la columna P2 del mes sale de las predicciones de TEST
        (`test_predictions_motor2.parquet`), es decir del snapshot con que se
        ENTRENÓ el modelo — un mes que ya pasó.

        Con este parámetro, la columna P2 del mes sale del scoring del mes en
        curso: es la lista accionable de "quién va a cancelar ahora". El nombre de
        la columna se deriva del snapshot que trae el archivo de scoring, no del
        corte, para que no quede un `p2_dec_2025` conteniendo datos de enero.

        Las columnas `p2_ultimo` y `p2_max` siguen saliendo del histórico de test:
        son la vista "estado reciente" y "peor momento del año", que el scoring de
        un solo mes no puede dar.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--p2-scoring", type=str, default=None)
    args, _ = p.parse_known_args()
    return args


def resolver_anio_p1(cutoff_year):
    """
    Año sobre el que se agrega P1. Normalmente el año del corte, pero si ese año
    todavía no tiene intentos de cobro cargados (caso típico: se corre en los
    primeros días de enero, cuando el año nuevo no tiene actividad), se usa el
    ÚLTIMO año con datos.

    Sin este fallback, correr con corte 2026-01-03 dejaba la tabla sin P1.
    """
    años = set()
    pf = pq.ParquetFile(MOTOR1_FEATURES)
    for batch in pf.iter_batches(columns=["attempt_ts"], batch_size=400_000):
        años.update(batch.to_pandas()["attempt_ts"].dt.year.unique().tolist())
    if not años:
        raise SystemExit(
            "\n[ERROR] motor1_features.parquet no tiene ningún intento de cobro.\n")
    if cutoff_year in años:
        return cutoff_year
    anio = max(a for a in años if a <= cutoff_year) if any(a <= cutoff_year for a in años) else max(años)
    print(f"  [aviso] El año del corte ({cutoff_year}) no tiene intentos cargados. "
          f"P1 se agrega sobre {anio} (último año con datos). Los nombres de las "
          f"columnas de comportamiento usan ese año.")
    return anio


def main():
    CUTOFF = get_cutoff_date()
    extra = parse_extra_args()
    CUTOFF_YEAR  = CUTOFF.year                        # ej: 2026
    CUTOFF_SLUG  = CUTOFF.strftime("%Y_%m_%d")        # ej: "2026_03_31"
    CUTOFF_MONTH = CUTOFF.replace(day=1)              # primer día del mes de corte
    TENURE_COL   = f"tenure_days_at_{CUTOFF_SLUG}"   # ej: "tenure_days_at_2026_03_31"

    # Año de agregación de P1 (con fallback al último año con datos).
    P1_YEAR      = resolver_anio_p1(CUTOFF_YEAR)
    N_INT_COL    = f"n_intentos_{P1_YEAR}"
    N_EX_COL     = f"n_exitos_{P1_YEAR}"
    TASA_COL     = f"tasa_exito_{P1_YEAR}"

    # Fuente y nombre de la columna P2 del mes.
    P2_SCORING = Path(extra.p2_scoring) if extra.p2_scoring else None
    if P2_SCORING and not P2_SCORING.exists():
        raise SystemExit(f"\n[ERROR] No existe el scoring de Motor 2: {P2_SCORING}\n")
    if P2_SCORING:
        snap_scoring = pd.to_datetime(
            pd.read_parquet(P2_SCORING, columns=["snapshot_ts"])["snapshot_ts"]).max()
        P2_SNAP_COL = f"p2_{snap_scoring.strftime('%b_%Y').lower()}"
        print(f"  [P2] columna del mes desde el scoring {P2_SCORING.name} "
              f"(snapshot {snap_scoring.date()}) → {P2_SNAP_COL}")
    else:
        snap_scoring = None
        P2_SNAP_COL = f"p2_{CUTOFF.strftime('%b_%Y').lower()}"  # ej: "p2_mar_2026"
        print(f"  [P2] columna del mes desde las predicciones de test → {P2_SNAP_COL}")

    print("[1/6] Cargando predicciones Motor 1 v4 + payment_id → (pledge_id, attempt_ts)...")
    # Las predicciones de Motor 1 vienen a nivel intento (1 fila por payment_id)
    # con la probabilidad de éxito predicha. Necesitamos enriquecer con pledge_id
    # y attempt_ts (que NO están en el archivo de predicciones) leyéndolos del
    # staging.
    m1 = pd.read_parquet(PRED_M1, columns=["payment_id", "is_success", "prob_success"])
    m1 = m1.rename(columns={"prob_success": "p1_intento"})
    print(f"  intentos: {len(m1):,}")

    # Lectura por batches del staging para no cargar 5.3M filas en RAM.
    # Filtramos a 2025 al vuelo: solo nos interesa el período de test del modelo.
    pf = pq.ParquetFile(MOTOR1_FEATURES)
    map_chunks = []
    for batch in pf.iter_batches(columns=["payment_id", "pledge_id", "attempt_ts"], batch_size=400_000):
        c = batch.to_pandas()
        c = c[c["attempt_ts"].dt.year == P1_YEAR]
        if len(c) == 0:
            continue
        map_chunks.append(c)
    if not map_chunks:
        # Con el fallback de resolver_anio_p1() esto no debería ocurrir: significa
        # que el año elegido quedó sin intentos entre lectura y lectura. Mensaje
        # explícito en vez del "No objects to concatenate" de pandas.
        raise SystemExit(
            f"\n[ERROR] No hay intentos de cobro del año {P1_YEAR} en "
            f"motor1_features.parquet, así que P1 no se puede calcular.\n"
            f"        Regenera las features de Motor 1 (etapas 0 y 1) y reintenta.\n")
    mp = pd.concat(map_chunks, ignore_index=True)
    del map_chunks
    # LEFT join: preserva las predicciones aunque algún payment_id no tenga
    # match en staging (no debería ocurrir, pero defensivo).
    m1 = m1.merge(mp, on="payment_id", how="left")
    del mp
    gc.collect()

    # === Agregación P1 por pledge ===
    # P1 representa la calidad esperada de cobro del donante. Calculamos:
    #   - p1_promedio: promedio sobre TODOS los intentos del año
    #     (vista "calidad típica del donante")
    #   - p1_ultimo_intento: P1 del último intento
    #     (vista "estado más reciente")
    #   - n_intentos_2025, n_exitos_2025, tasa_exito_2025:
    #     comportamiento REAL observado vs lo que predijo el modelo
    #     (permite calibración por donante)
    #   - primer_attempt_ts, ultimo_attempt_ts: span de actividad del año
    #
    # Ordenamiento previo por (pledge_id, attempt_ts) garantiza que
    # `.agg("last")` tome el cronológicamente último, no un orden aleatorio.
    print("[2/6] Agregando P1 a nivel pledge...")
    m1 = m1.sort_values(["pledge_id", "attempt_ts"])
    p1_by_pledge = m1.groupby("pledge_id").agg(
        p1_promedio=("p1_intento", "mean"),
        p1_ultimo_intento=("p1_intento", "last"),
        n_intentos_tmp=("p1_intento", "size"),
        ultimo_attempt_ts=("attempt_ts", "max"),
        primer_attempt_ts=("attempt_ts", "min"),
        n_exitos_tmp=("is_success", "sum"),
    ).reset_index()
    p1_by_pledge = p1_by_pledge.rename(columns={
        "n_intentos_tmp": N_INT_COL,
        "n_exitos_tmp":   N_EX_COL,
    })
    p1_by_pledge[TASA_COL] = (p1_by_pledge[N_EX_COL] / p1_by_pledge[N_INT_COL]).astype("float32")
    print(f"  pledges con intentos en {P1_YEAR}: {len(p1_by_pledge):,}")
    del m1
    gc.collect()

    print("[3/6] Cargando predicciones Motor 2 (snapshots pledge×mes del histórico)...")
    m2 = pd.read_parquet(PRED_M2)
    m2 = m2.rename(columns={"prob_churn": "p2_snapshot"})
    m2["snapshot_ts"] = pd.to_datetime(m2["snapshot_ts"])
    print(f"  snapshots: {len(m2):,}  "
          f"({m2['snapshot_ts'].min():%Y-%m} → {m2['snapshot_ts'].max():%Y-%m})")

    # === Agregación P2 por pledge ===
    # Tres vistas distintas de P2 por pledge:
    #   - p2_ultimo: P2 del snapshot MÁS RECIENTE del histórico
    #     → estado del riesgo al cierre del período evaluado
    #   - p2_max: P2 máximo del histórico
    #     → peor momento (vista conservadora)
    #   - p2_<mmm>_<aaaa>: el snapshot puntual accionable. Sale del scoring
    #     mensual si se pasó --p2-scoring (mes en curso, lo que el equipo de
    #     retención debe llamar AHORA); si no, del snapshot del mes de corte
    #     dentro del histórico (un mes que ya pasó, útil para validar el modelo
    #     contra cancelaciones reales).
    print("[4/6] Agregando P2 a nivel pledge...")
    p2_last = m2.sort_values(["pledge_id", "snapshot_ts"]).groupby("pledge_id").agg(
        p2_ultimo=("p2_snapshot", "last"),
        ultimo_snapshot_ts=("snapshot_ts", "max"),
        p2_max=("p2_snapshot", "max"),
    ).reset_index()

    if P2_SCORING:
        p2_snap = pd.read_parquet(P2_SCORING, columns=["pledge_id", "snapshot_ts", "prob_churn"])
        p2_snap["snapshot_ts"] = pd.to_datetime(p2_snap["snapshot_ts"])
        p2_snap = p2_snap[p2_snap["snapshot_ts"] == snap_scoring]
        p2_snap = p2_snap[["pledge_id", "prob_churn"]].rename(columns={"prob_churn": P2_SNAP_COL})
        etiqueta_snap = f"{snap_scoring:%b-%Y} (scoring)"
    else:
        # Snapshot del mes de corte dentro del histórico (puede no existir para
        # pledges que churnearon antes — esos quedan en NA).
        p2_snap = m2[m2["snapshot_ts"] == CUTOFF_MONTH][["pledge_id", "p2_snapshot"]].rename(
            columns={"p2_snapshot": P2_SNAP_COL}
        )
        etiqueta_snap = f"{CUTOFF_MONTH:%b-%Y} (test)"

    p2_by_pledge = p2_last.merge(p2_snap, on="pledge_id", how="left")
    print(f"  pledges con snapshots en el histórico: {len(p2_by_pledge):,}")
    print(f"  pledges con {P2_SNAP_COL} — {etiqueta_snap}: "
          f"{p2_by_pledge[P2_SNAP_COL].notna().sum():,}")
    del m2, p2_last, p2_snap
    gc.collect()

    print("[5/6] Cargando atributos del pledge (último estado conocido + Pledge metadata)...")
    # Atributos del pledge: del staging_v3 tomamos el último intento de 2025 como
    # "snapshot de atributos". Para el detalle del contrato (start, end, status), Pledge.
    cols_attr = ["pledge_id", "contact_id", "attempt_ts", "payment_amount", "payment_method",
                 "payment_gateway", "payment_submethod", "bank_canonical",
                 "state_norm", "gender_norm", "age_band", "ses_level",
                 "pre_failure_streak", "success_count_cum", "failure_count_cum",
                 "contact_age_days"]
    cols_keep = cols_attr  # v8: se incluye payment_method (medio de pago)
    pf = pq.ParquetFile(MOTOR1_FEATURES)
    attrs_chunks = []
    for batch in pf.iter_batches(columns=cols_keep, batch_size=400_000):
        c = batch.to_pandas()
        c = c[c["attempt_ts"].dt.year == P1_YEAR]
        if len(c) == 0:
            continue
        attrs_chunks.append(c)
    attrs = pd.concat(attrs_chunks, ignore_index=True)
    del attrs_chunks
    gc.collect()
    # último intento por pledge
    attrs = attrs.sort_values(["pledge_id", "attempt_ts"]).drop_duplicates("pledge_id", keep="last")
    attrs["payment_amount"] = pd.to_numeric(
        attrs["payment_amount"].astype(str).str.replace(r"[^\d\.]","",regex=True),
        errors="coerce"
    ).fillna(0).astype("float32")
    print(f"  atributos por pledge: {len(attrs):,}")

    # Allocation features (último estado al fin de 2025)
    print("  Cruzando con Allocation features...")
    af = pd.read_parquet(ALLOC_FEAT, columns=["payment_id", "alloc_n_total",
                                                "alloc_n_downgrades_total",
                                                "alloc_recent90_downgrade",
                                                "alloc_last_type",
                                                "alloc_days_since_last"])
    # Necesitamos el alloc del último intento del pledge — leemos payment_id del staging
    pf2 = pq.ParquetFile(MOTOR1_FEATURES)
    pid_chunks = []
    for batch in pf2.iter_batches(columns=["payment_id", "pledge_id", "attempt_ts"], batch_size=400_000):
        c = batch.to_pandas()
        c = c[c["attempt_ts"].dt.year == P1_YEAR]
        if len(c) == 0:
            continue
        pid_chunks.append(c)
    pids = pd.concat(pid_chunks, ignore_index=True)
    del pid_chunks
    pids = pids.sort_values(["pledge_id", "attempt_ts"]).drop_duplicates("pledge_id", keep="last")
    pids = pids.merge(af, on="payment_id", how="left")
    pids = pids[["pledge_id", "alloc_n_total", "alloc_n_downgrades_total",
                 "alloc_recent90_downgrade", "alloc_last_type", "alloc_days_since_last"]]
    attrs = attrs.merge(pids, on="pledge_id", how="left")
    del af, pids
    gc.collect()

    # Pledge contractual data
    pl = pd.read_parquet(PLEDGE, columns=["id", "npsp_status_c", "npsp_start_date_c", "npsp_end_date_c"])
    pl = pl.rename(columns={"id": "pledge_id",
                             "npsp_status_c": "pledge_status",
                             "npsp_start_date_c": "pledge_start_date",
                             "npsp_end_date_c": "pledge_end_date"})
    pl["pledge_start_date"] = pd.to_datetime(pl["pledge_start_date"], errors="coerce")
    pl["pledge_end_date"] = pd.to_datetime(pl["pledge_end_date"], errors="coerce")
    attrs = attrs.merge(pl, on="pledge_id", how="left")

    # === Primer intento observado por pledge (TODA la historia) ===
    # Se recorre motor1_features SIN filtrar por año para obtener el min
    # attempt_ts real de cada pledge. Es la evidencia observada de cuándo el
    # donante empezó a pagar.
    pf_fa = pq.ParquetFile(MOTOR1_FEATURES)
    fa_chunks = []
    for batch in pf_fa.iter_batches(columns=["pledge_id", "attempt_ts"], batch_size=500_000):
        c = batch.to_pandas()
        fa_chunks.append(c.groupby("pledge_id", sort=False)["attempt_ts"].min())
    first_attempt_all = (pd.concat(fa_chunks).groupby(level=0).min()
                         .rename("first_attempt_all").reset_index())
    del fa_chunks
    gc.collect()
    attrs = attrs.merge(first_attempt_all, on="pledge_id", how="left")

    # === tenure_days_at_2025_12_31 (v8, fix de días negativos) ===
    # Antes: (2025-12-31 - pledge_start_date). Problema: npsp__Start_Date__c es
    # una fecha futura (2026) en ~1.45% de pledges (fecha de ciclo/renovación,
    # NO el inicio histórico), lo que producía tenure NEGATIVO aunque el donante
    # llevaba todo el año pagando.
    # Fix: anclar al inicio EFECTIVO = min(pledge_start_date, primer intento real).
    #   - Caso normal        → domina pledge_start_date (sin cambio de comportamiento).
    #   - Caso start futura   → domina el primer intento observado (la verdad del dato).
    # Como los datos terminan en 2025, el resultado queda siempre ≥ 0 sin
    # inventar valores ni descartar donantes activos.
    effective_start = attrs[["pledge_start_date", "first_attempt_all"]].min(axis=1)
    attrs[TENURE_COL] = (
        (CUTOFF - effective_start).dt.days
    ).astype("Int32")
    n_neg = int((attrs[TENURE_COL] < 0).sum())
    print(f"  tenure_days: min={attrs[TENURE_COL].min()}, "
          f"negativos={n_neg} (esperado 0)")
    attrs = attrs.drop(columns=["first_attempt_all"])

    # === Segmentación: Tipo de donante ===
    # Prospecting (< 12 meses): donantes nuevos en proceso de fidelización.
    # House (>= 12 meses): donantes consolidados / recurrentes establecidos.
    # Desconocido: pledges sin tenure poblado (raro pero posible).
    #

    attrs["tipo_donante"] = np.where(
        attrs[TENURE_COL] < 365, "Prospecting",
        np.where(attrs[TENURE_COL] >= 365, "House", "Desconocido")
    )
    del pl
    gc.collect()

    print("[6/6] Ensamblando tabla final por pledge...")
    # INNER join con p1: SOLO incluimos pledges que tuvieron al menos un
    # intento en 2025 (universo del modelo). Pledges 100% inactivos en
    # 2025 quedan fuera del entregable.
    final = attrs.merge(p1_by_pledge, on="pledge_id", how="inner")
    # LEFT join con p2: si un pledge no tiene snapshots en 2025 (raro pero
    # posible para pledges muy nuevos o churneados temprano), igual queda
    # en la tabla con P2=NA.
    final = final.merge(p2_by_pledge, on="pledge_id", how="left")

    # === Timing óptimo del próximo cobro (salida de 03_timing_reintento_motor1) ===
    # Agrega, por donante, CUÁNTOS DÍAS esperar y la FECHA recomendada para el
    # siguiente intento de cobro (el día que maximiza la probabilidad de éxito),
    # junto con esa probabilidad. LEFT join: si el timing aún no se generó, la
    # tabla se arma igual con estas columnas en NA (y se avisa).
    if TIMING.exists():
        tim = pd.read_parquet(TIMING, columns=[
            "pledge_id", "dias_esperar_desde_hoy", "fecha_recomendada", "prob_optimo"
        ]).rename(columns={
            "dias_esperar_desde_hoy": "dias_proximo_cobro",
            "fecha_recomendada":      "fecha_proximo_cobro",
            "prob_optimo":            "prob_exito_proximo_cobro",
        })
        final = final.merge(tim, on="pledge_id", how="left")
        n_tim = int(final["fecha_proximo_cobro"].notna().sum())
        print(f"  timing próximo cobro: {n_tim:,} pledges con recomendación "
              f"({100*n_tim/len(final):.1f}%)")
    else:
        print("  [aviso] no existe results/motor1/timing_reintento_motor1.parquet — "
              "las columnas de próximo cobro quedarán vacías. "
              "Corré 03_timing_reintento_motor1.py antes para poblarlas.")
        final["dias_proximo_cobro"] = pd.NA
        final["fecha_proximo_cobro"] = pd.NA
        final["prob_exito_proximo_cobro"] = pd.NA

    # Renombre y ordenamiento ejecutivo
    cols_order = [
        "pledge_id", "contact_id",
        "pledge_status", "pledge_start_date", "pledge_end_date",
        TENURE_COL, "tipo_donante",
        # P1 / P2
        "p1_promedio", "p1_ultimo_intento", "p2_ultimo", "p2_max", P2_SNAP_COL,
        # Próximo cobro recomendado (timing óptimo, Motor 1)
        "dias_proximo_cobro", "fecha_proximo_cobro", "prob_exito_proximo_cobro",
        # Comportamiento histórico
        N_INT_COL, N_EX_COL, TASA_COL,
        "pre_failure_streak", "failure_count_cum", "success_count_cum",
        # Atributos del último intento (snapshot)
        "ultimo_attempt_ts", "payment_amount", "payment_gateway",
        "payment_method", "payment_submethod", "bank_canonical",
        # Demografía
        "state_norm", "gender_norm", "age_band", "ses_level", "contact_age_days",
        # Allocation
        "alloc_n_total", "alloc_n_downgrades_total", "alloc_recent90_downgrade",
        "alloc_last_type", "alloc_days_since_last",
        # Snapshots de referencia
        "primer_attempt_ts", "ultimo_snapshot_ts",
    ]
    final = final[[c for c in cols_order if c in final.columns]]

    # Limpiar tipos para CSV legible
    for c in ["p1_promedio", "p1_ultimo_intento", "p2_ultimo", "p2_max", P2_SNAP_COL,
              "prob_exito_proximo_cobro"]:
        if c in final.columns:
            final[c] = final[c].astype("float32").round(4)

    # Stats
    n = len(final)
    n_p2_dic = final[P2_SNAP_COL].notna().sum() if P2_SNAP_COL in final.columns else 0
    print(f"\n  Total filas: {n:,}")
    print(f"  Con {P2_SNAP_COL} — {etiqueta_snap}: {n_p2_dic:,}")
    print(f"  Distribución tipo_donante:")
    print(final["tipo_donante"].value_counts())
    print(f"  Distribución pledge_status:")
    print(final["pledge_status"].value_counts())

    # Sanity checks
    print(f"\n  Stats P1: media={final['p1_promedio'].mean():.3f}, mediana={final['p1_promedio'].median():.3f}")
    print(f"  Stats P2 (último): media={final['p2_ultimo'].mean():.3f}, p95={final['p2_ultimo'].quantile(0.95):.3f}")
    snap_label = etiqueta_snap
    print(f"  Stats P2 ({snap_label}): n={n_p2_dic:,} | "
          f"P2>80% = {(final[P2_SNAP_COL] > 0.8).sum():,} pledges (alta alerta)")
    print(f"  Stats P2 ({snap_label}): P2>60% = {(final[P2_SNAP_COL] > 0.6).sum():,} pledges (alerta)")

    # Guardar — nombre incluye la fecha de corte para trazabilidad
    out_name = f"Tabla_Donantes_P1_P2_{CUTOFF_SLUG}"
    csv_path = OUT_DIR / f"{out_name}.csv"
    parquet_path = OUT_DIR / f"{out_name}.parquet"
    final.to_csv(csv_path, index=False)
    final.to_parquet(parquet_path, index=False)
    print(f"\n[OK] {csv_path}")
    print(f"[OK] {parquet_path}")
    print(f"  Tamaño CSV: {csv_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
