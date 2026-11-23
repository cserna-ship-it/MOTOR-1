#!/usr/bin/env python3
"""
02_motor2_dataset.py (v2) — Features Motor 2 (churn 30 días)

PROPÓSITO
---------
Construir el dataset para Motor 2: predicción de probabilidad de que un
pledge cancele (churnee) en los 30 días siguientes a una fecha de
observación (snapshot).

GRANULARIDAD: PLEDGE × SNAPSHOT MENSUAL
---------------------------------------
Mientras Motor 1 trabaja con UNA FILA POR INTENTO, Motor 2 trabaja con
UNA FILA POR (pledge, mes-snapshot). Por cada pledge, generamos hasta
24 filas (una por cada mes entre 2024-01 y 2025-12), cada una
representando el estado del pledge al inicio de ese mes.

Por ejemplo:
    Pledge X (activo desde 2023-05, sigue activo a fin de 2025):
        24 filas: (X, 2024-01-01), (X, 2024-02-01), ..., (X, 2025-12-01)

    Pledge Y (activo desde 2024-06, churneó en 2024-11-15):
        6 filas: (Y, 2024-06-01), (Y, 2024-07-01), ..., (Y, 2024-11-01)
        La fila de 2024-11-01 tiene label_churn_30d=1 (cancela dentro de 30d).
        Las anteriores tienen label=0.
        Snapshots posteriores (2024-12-01+) SE EXCLUYEN porque el pledge
        ya no existe.

REGLAS DE NEGOCIO
-----------------
1. **Ventana global de snapshots**: 24 meses terminando en el mes de la
   fecha de corte (primer día de cada mes). Con el corte por defecto
   (2025-12-31) → 2024-01-01 a 2025-12-01. Con `--cutoff-date 2026-01-03`
   → 2024-02-01 a 2026-01-01. 24 snapshots posibles por pledge.

2. **Inicio efectivo del pledge** (`effective_start`):
       - Si `npsp__StartDate__c` existe → usarlo.
       - Si no → usar el primer attempt_ts del pledge en payments.
   Esto cubre pledges con metadata incompleta.

3. **Snapshots se descartan si**:
       - snapshot_ts < effective_start (el pledge no existía aún)
       - snapshot_ts > npsp_end_date_c (el pledge ya churneó)

4. **Label de churn** (`label_churn_30d`):
   1 si `(snapshot_ts, snapshot_ts + 30d]` contiene la fecha de fin
   del pledge (`npsp_end_date_c`), 0 si no.
   El label depende EXCLUSIVAMENTE de `npsp__EndDate__c` (fecha de
   cancelación contractual del pledge). NO depende de payment_status.
   Un pledge puede fallar muchos cobros consecutivos sin churnear si
   el donante no cancela formalmente.

5. **Anti-leakage**: TODAS las features se calculan con información
   ESTRICTAMENTE anterior al snapshot. Se usa `direction="backward"` y
   `allow_exact_matches=False` en los merge_asof.

FEATURES (17)
-------------
Del último intento previo al snapshot:
    days_since_last_attempt, last_attempt_success, last_payment_amount,
    last_payment_method, last_bank_name, last_gateway

Conteos acumulados (todo el histórico):
    cum_intentos, cum_exitos, cum_fallos, success_rate_hist

Ventana de 30 días previos al snapshot:
    n_intentos_30d, n_exitos_30d, n_fallos_30d

Estado de racha y antigüedad:
    streak_actual, tenure_days

MANEJO DE MÚLTIPLES INTENTOS POR MES
------------------------------------
Si un pledge tiene varios intentos en un mes:
    - NO se descarta ninguno. Los conteos (n_intentos_30d, cum_intentos)
      cuentan TODOS.
    - Las features `last_*` toman el MÁS RECIENTE de los intentos previos
      al snapshot.

ENTRADA / SALIDA
----------------
Lee:    data/staging/stg_payment_attempts.parquet
        data/unified/pledge/pledge_unified.parquet
Genera: data/features/motor2_pledge_month.parquet
        (2.5M filas × 18 cols con el corte por defecto; el número de filas
         cambia al mover la fecha de corte)

CAMBIO (v9 — Motor 2 a nivel donante, borrador)
------------------------------------------------
Se agrega la columna `contact_id` a la salida (18 → 19 cols). Ya existía en
el staging (join con Opportunity hecho en 01_create_staging.py); antes no se
leía porque Motor 2 solo necesitaba pledge_id. Se agrega ahora para que
01b_motor2_dataset_donante.py pueda agrupar pledges del mismo donante sin
tener que releer el staging por su cuenta.

IMPORTANTE: este cambio ROMPE la verificación de columnas de run_all.sh
(`verify_step "motor2_pledge_month" ... 2548075 18` espera 18 columnas, no
19). Hay que actualizar ese número en run_all.sh antes de correr el pipeline
completo con este script modificado — ver bitácora del cambio para el
detalle completo.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/ al path
from config import get_cutoff_date
STAGING = ROOT / "data/staging/stg_payment_attempts.parquet"
PLEDGE = ROOT / "data/unified/pledge/pledge_unified.parquet"
OUT = ROOT / "data/features"
OUT.mkdir(parents=True, exist_ok=True)


def parse_extra_args():
    """
    Parámetros propios de este script (la fecha de corte la resuelve config.py).

    --months N   Cuántos snapshots mensuales generar, contando hacia atrás desde
                 el mes de corte. Default 24 (dataset de entrenamiento).
                 `--months 1` genera SOLO el snapshot del mes de corte, que es lo
                 que necesita la puntuación mensual (04_score_motor2.py).

                 Importante: el número de meses NO afecta el valor de las
                 features. Los acumulados (cum_*, streak_actual, tenure_days) se
                 calculan siempre sobre TODA la historia de intentos disponible;
                 la ventana solo decide CUÁLES snapshots se emiten.

    --out RUTA   Parquet de salida. Default data/features/motor2_pledge_month.parquet.
                 Se usa para no sobreescribir el dataset de entrenamiento cuando
                 solo se está generando el snapshot del mes.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--months", type=int, default=24)
    p.add_argument("--out", type=str, default=None)
    args, _ = p.parse_known_args()
    if args.months < 1:
        raise SystemExit(f"[ERROR] --months debe ser >= 1 (recibido: {args.months})")
    return args


def main():
    CUTOFF = get_cutoff_date()
    extra = parse_extra_args()
    N_MESES = extra.months
    OUT_PATH = Path(extra.out) if extra.out else OUT / "motor2_pledge_month.parquet"
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ventana de snapshots: N_MESES terminando en el mes de corte.
    # Ej: cutoff=2026-03-31, months=24 → GLOBAL_END=2026-03-01, GLOBAL_START=2024-04-01
    #     cutoff=2026-03-31, months=1  → un solo snapshot: 2026-03-01
    GLOBAL_END = CUTOFF.replace(day=1)
    GLOBAL_START = (GLOBAL_END - pd.DateOffset(months=N_MESES - 1)).replace(day=1)
    print(f"  Ventana de snapshots: {GLOBAL_START:%Y-%m} → {GLOBAL_END:%Y-%m} ({N_MESES} mes(es))")
    print(f"  Salida: {OUT_PATH}")

    print("[1/5] Cargando pledges y staging...")
    # Staging slim: solo las columnas necesarias para las features.
    # CAMBIO (v9 — Motor 2 a nivel donante): se agrega "contact_id". Ya existe
    # en el staging (viene del join con Opportunity en 01_create_staging.py),
    # solo no se estaba leyendo acá. Se necesita para poder agrupar pledges
    # del mismo donante aguas abajo (01b_motor2_dataset_donante.py), sin que
    # cada consumidor tenga que volver a leer el staging completo por su cuenta.
    cols_stg = ["pledge_id", "attempt_ts", "is_success", "contact_id",
                "payment_amount", "payment_method", "bank_name", "payment_gateway"]
    stg = pd.read_parquet(STAGING, columns=cols_stg)
    stg["is_success"] = stg["is_success"].astype("int8")
    stg["attempt_ts"] = stg["attempt_ts"].astype("datetime64[ns]")
    # payment_amount viene como string con caracteres no numéricos en algunos
    # registros (símbolos de moneda, comas, etc.). Limpiamos antes de convertir.
    stg["payment_amount"] = pd.to_numeric(
        stg["payment_amount"].astype(str).str.replace(r"[^\d\.]", "", regex=True),
        errors="coerce"
    ).fillna(0).astype("float32")

    # Pledge: necesitamos start_date (para tenure y filtro de snapshots),
    # end_date (para LABEL DE CHURN), y status (para validación opcional).
    # Mapeo a campos del CSV:
    #   id                ← "Id"
    #   npsp_start_date_c ← "npsp__StartDate__c"
    #   npsp_end_date_c   ← "npsp__EndDate__c"   ← define el target
    #   npsp_status_c     ← "npsp__Status__c"
    pl = pd.read_parquet(PLEDGE, columns=["id", "npsp_start_date_c", "npsp_end_date_c", "npsp_status_c"])
    pl = pl.rename(columns={"id": "pledge_id"})
    pl["npsp_start_date_c"] = pd.to_datetime(pl["npsp_start_date_c"], errors="coerce")
    pl["npsp_end_date_c"] = pd.to_datetime(pl["npsp_end_date_c"], errors="coerce")
    print(f"  staging filas: {len(stg):,} | pledges: {pl['pledge_id'].nunique():,}")

    # === Coherencia corte ↔ horizonte de los datos ===
    # Las features de cada snapshot se calculan con los intentos ANTERIORES a
    # ese snapshot. Si el corte va muy por delante del último intento cargado,
    # los snapshots finales se construyen sobre datos que no existen: quedan con
    # days_since_last_attempt inflado y conteos congelados. Se avisa (no se
    # aborta: puede ser intencional al simular un corte futuro).
    max_attempt = stg["attempt_ts"].max()
    print(f"  Último intento en staging: {max_attempt:%Y-%m-%d}")

    # === Guardarraíl de frescura de datos ===
    # Las features se miden DESDE el snapshot hacia atrás. Si el staging se
    # quedó meses atrás del snapshot, "días desde el último intento" le da a
    # TODO el mundo un número enorme y el modelo concluye que la base entera
    # está muerta: el P2 medio se dispara y la lista de retención no sirve.
    #
    # Caso real observado: con el staging hasta 2025-12-31 y snapshot 2026-09-01,
    # la mediana de días sin intento pasa de 31 a 274 y el P2 medio de 0,03 a 0,27.
    #
    # run_monthly.sh ya bloquea esto (etapa C regenera el staging y aborta si no
    # alcanza). Este aviso cubre la ejecución manual del script, que no pasa por ahí.
    dias_rezago = (GLOBAL_END - max_attempt).days
    if dias_rezago > 45:
        print(f"  [AVISO] El snapshot más reciente ({GLOBAL_END:%Y-%m-%d}) va "
              f"{dias_rezago} días por delante del último pago en el staging "
              f"({max_attempt:%Y-%m-%d}).")
        print(f"          Las features de actividad reciente van a salir infladas y el "
              f"riesgo va a parecer altísimo para toda la base.")
        print(f"          Carga los pagos que faltan en raw/ y regenera el staging, o "
              f"usa un corte más cercano a {max_attempt:%Y-%m-%d}.")
    if GLOBAL_END > max_attempt:
        meses_sin_datos = ((GLOBAL_END.year - max_attempt.year) * 12
                           + GLOBAL_END.month - max_attempt.month)
        print(f"  [AVISO] El mes de corte ({GLOBAL_END:%Y-%m}) va {meses_sin_datos} mes(es) "
              f"por delante del último intento cargado ({max_attempt:%Y-%m-%d}). "
              f"Los snapshots posteriores a esa fecha se calculan sin intentos nuevos. "
              f"Si esperabas datos más recientes, actualiza raw/ y re-corre "
              f"00_unify_raw_data.py + 01_create_staging.py.")

    # === Inicio efectivo del pledge ===
    # Regla: si npsp_start_date_c está poblado → usarlo.
    # Si no → usar el primer attempt_ts observado del pledge.
    # Esto cubre pledges con metadata incompleta en el CRM (~5% de casos).
    print("[2/5] Determinando ventana activa por pledge...")
    first_attempt = stg.groupby("pledge_id", observed=True)["attempt_ts"].min().rename("first_attempt")
    pl = pl.merge(first_attempt, on="pledge_id", how="left")
    pl["effective_start"] = pl["npsp_start_date_c"].fillna(pl["first_attempt"])

    # Snapshot dates por pledge: vectorizado generando lista
    print("[3/5] Generando snapshots mensuales (vectorizado)...")
    months = pd.date_range(GLOBAL_START, GLOBAL_END, freq="MS")
    print(f"  Meses considerados: {len(months)}")

    # Construir cross product limitado por start/end
    pl_simple = pl[["pledge_id", "effective_start", "npsp_end_date_c", "npsp_status_c"]].dropna(subset=["effective_start"])
    print(f"  Pledges con start válido: {len(pl_simple):,}")

    # Repetir cada pledge × meses (broadcast)
    pids = pl_simple["pledge_id"].values
    starts = pl_simple["effective_start"].values
    ends = pl_simple["npsp_end_date_c"].values  # NaT permitido
    n_p = len(pids)
    n_m = len(months)

    # Repetir
    pid_rep = np.repeat(pids, n_m)
    start_rep = np.repeat(starts, n_m)
    end_rep = np.repeat(ends, n_m)
    snap_rep = np.tile(months.values, n_p)

    snap = pd.DataFrame({
        "pledge_id": pid_rep,
        "snapshot_ts": pd.to_datetime(snap_rep).astype("datetime64[ns]"),
        "effective_start": pd.to_datetime(start_rep).astype("datetime64[ns]"),
        "npsp_end_date_c": pd.to_datetime(end_rep).astype("datetime64[ns]"),
    })
    print(f"  Filas cross-product brutas: {len(snap):,}")

    # === Filtrado de snapshots inválidos ===
    # Dos reglas combinadas:
    #   a) snapshot_ts >= effective_start → el pledge ya existía
    #   b) Si hay npsp_end_date_c: snapshot_ts <= end → el pledge no había
    #      churneado todavía. Si end es NaT (pledge aún activo): sin restricción.
    keep = (snap["snapshot_ts"] >= snap["effective_start"])
    has_end = snap["npsp_end_date_c"].notna()
    keep = keep & (~has_end | (snap["snapshot_ts"] <= snap["npsp_end_date_c"]))
    snap = snap[keep].reset_index(drop=True)
    print(f"  Snapshots válidos: {len(snap):,}")

    # === LABEL DE CHURN ===
    # label_churn_30d = 1 ↔ el pledge cancela en el rango (snapshot, snapshot+30d]
    # (intervalo abierto a la izquierda, cerrado a la derecha).
    #
    # Construcción:
    #   days_to_end = días entre el snapshot y npsp_end_date_c
    #   1 si 1 <= days_to_end <= 30
    #   0 en cualquier otro caso (incluido NaT en end_date → pledge sigue activo)
    #
    # NOTA: el `1 <= days_to_end` (no `0 <`) es deliberado para evitar
    # casos donde end_date == snapshot_ts (eventos exactos en el snapshot
    # se tratan como "ya churneó", no se predicen).
    days_to_end = (snap["npsp_end_date_c"] - snap["snapshot_ts"]).dt.days
    snap["label_churn_30d"] = ((days_to_end >= 1) & (days_to_end <= 30)).astype("int8")
    # Si el pledge no tiene end_date (sigue activo) → label = 0 forzadamente.
    snap["label_churn_30d"] = snap["label_churn_30d"].where(snap["npsp_end_date_c"].notna(), 0).astype("int8")
    print(f"  Tasa churn 30d global: {snap['label_churn_30d'].mean():.4f}")

    # ----------- Features por snapshot via merge_asof y agregaciones -----------
    print("[4/5] Calculando features con merge_asof y groupby (esto puede tardar 1-2 min)...")
    stg_sorted = stg.sort_values("attempt_ts")
    snap_sorted = snap.sort_values("snapshot_ts").reset_index(drop=True)

    # Para cada snapshot, queremos: último intento antes, n_intentos en últimos 30/90 días, fallos, streak.
    # Estrategia: para cada pledge, hacer merge_asof "last attempt before snapshot".
    # Y para los conteos: hacer una "expansión" de intentos previos.

    # Method: agrupar staging por pledge, y para cada pledge mantener un dataframe ordenado.
    # Luego para cada snapshot del pledge, hacer cálculos vectorizados sobre el array de intentos previos.

    # Para vectorizar bien usamos un trick: convertimos staging a indexado y mergeamos contra snapshots con merge_asof por pledge_id.
    # Pero merge_asof necesita ordenamiento global por la "on" key. Lo hacemos con "by".

    snap_sorted = snap_sorted.sort_values(["snapshot_ts"]).reset_index(drop=True)
    stg_sorted = stg_sorted.sort_values(["attempt_ts"]).reset_index(drop=True)

    # === 1. Último intento antes del snapshot ===
    # Si un pledge tiene N intentos previos al snapshot, este merge_asof
    # toma SOLO el más reciente. Los otros N-1 NO entran a las features
    # last_*, pero SÍ entran a los conteos (cum_intentos, n_intentos_30d).
    #
    # Parámetros críticos (anti-leakage):
    #   - direction="backward" → buscar el ANTERIOR al snapshot
    #   - allow_exact_matches=False → un intento exactamente en snapshot_ts
    #     NO se considera (entraría al siguiente snapshot)
    asof_last = pd.merge_asof(
        snap_sorted[["pledge_id", "snapshot_ts"]],
        stg_sorted[["pledge_id", "attempt_ts", "is_success", "payment_amount", "payment_method", "bank_name", "payment_gateway"]],
        left_on="snapshot_ts", right_on="attempt_ts", by="pledge_id",
        direction="backward", allow_exact_matches=False
    )
    asof_last["days_since_last_attempt"] = (asof_last["snapshot_ts"] - asof_last["attempt_ts"]).dt.days.astype("float32")
    asof_last = asof_last.rename(columns={"is_success": "last_attempt_success",
                                          "payment_amount": "last_payment_amount",
                                          "payment_method": "last_payment_method",
                                          "bank_name": "last_bank_name",
                                          "payment_gateway": "last_gateway"})
    print(f"  asof OK: {asof_last.shape}")

    # 2. Conteos por ventanas: hacemos un join "all" filtrado por ventana de tiempo.
    # Para no explotar memoria, lo hacemos por pledge en chunks.
    print("  Calculando agregados por ventana...")

    # === 2. Pre-agregación mensual por pledge ===
    # En vez de iterar intento por intento, agregamos los intentos a nivel
    # (pledge, mes) UNA VEZ. Luego los conteos por ventana de tiempo se
    # resuelven con merge_asof sobre esta tabla más chica (~200x menos filas).
    # Es la optimización clave que permite procesar 5.3M intentos en pocos
    # minutos.
    stg_sorted["mes_intento"] = stg_sorted["attempt_ts"].values.astype("datetime64[M]")
    agg_mes = stg_sorted.groupby(["pledge_id", "mes_intento"], observed=True).agg(
        n_intentos=("is_success", "size"),
        n_exitos=("is_success", "sum"),
        sum_monto=("payment_amount", "sum"),
    ).reset_index()
    agg_mes["n_fallos"] = agg_mes["n_intentos"] - agg_mes["n_exitos"]
    print(f"  agg_mes filas: {len(agg_mes):,}")

    # Para cada snapshot, sumar agg_mes donde mes_intento < snapshot (cumulativos hasta el mes anterior)
    # y agg_mes donde mes_intento ∈ {snap-1m} para últimos 30d aprox.
    agg_mes = agg_mes.sort_values(["pledge_id", "mes_intento"])
    agg_mes["cum_intentos"] = agg_mes.groupby("pledge_id", observed=True)["n_intentos"].cumsum()
    agg_mes["cum_exitos"] = agg_mes.groupby("pledge_id", observed=True)["n_exitos"].cumsum()
    agg_mes["cum_fallos"] = agg_mes.groupby("pledge_id", observed=True)["n_fallos"].cumsum()

    # Para cada snapshot: tomar el agg_mes más reciente con mes_intento < snapshot (asof)
    snap_for_join = snap_sorted[["pledge_id", "snapshot_ts"]].copy()
    snap_for_join["snap_month"] = snap_for_join["snapshot_ts"].values.astype("datetime64[M]")

    # asof por mes: ordenar
    snap_for_join = snap_for_join.sort_values(["snap_month"]).reset_index(drop=True)
    agg_mes_sorted = agg_mes.sort_values(["mes_intento"]).reset_index(drop=True)
    cum = pd.merge_asof(
        snap_for_join,
        agg_mes_sorted[["pledge_id", "mes_intento", "cum_intentos", "cum_exitos", "cum_fallos"]],
        left_on="snap_month", right_on="mes_intento", by="pledge_id",
        direction="backward", allow_exact_matches=False
    )
    cum["cum_intentos"] = cum["cum_intentos"].fillna(0).astype("int32")
    cum["cum_exitos"] = cum["cum_exitos"].fillna(0).astype("int32")
    cum["cum_fallos"] = cum["cum_fallos"].fillna(0).astype("int32")
    cum["success_rate_hist"] = (cum["cum_exitos"] / cum["cum_intentos"].replace(0, np.nan)).fillna(0).astype("float32")
    print(f"  cum OK: {cum.shape}")

    # Últimos 30 días: usar agg_mes solo del mes inmediatamente anterior al snapshot
    # (aproximación; suficiente para un baseline de features)
    cum["prev_month"] = (cum["snap_month"].astype("datetime64[ns]") - pd.offsets.MonthBegin(1)).values.astype("datetime64[M]")
    last_m = agg_mes.merge(cum[["pledge_id", "snap_month", "prev_month"]], left_on=["pledge_id", "mes_intento"], right_on=["pledge_id", "prev_month"], how="right")
    last_m = last_m.rename(columns={"n_intentos": "n_intentos_30d", "n_exitos": "n_exitos_30d", "n_fallos": "n_fallos_30d"})
    last_m[["n_intentos_30d", "n_exitos_30d", "n_fallos_30d"]] = last_m[["n_intentos_30d", "n_exitos_30d", "n_fallos_30d"]].fillna(0).astype("int32")
    last_m = last_m[["pledge_id", "snap_month", "n_intentos_30d", "n_exitos_30d", "n_fallos_30d"]]

    # Streak actual: consecutivos al final del array de intentos previos al snapshot
    # Vectorizable parcialmente. Hacemos por pledge usando groupby.
    print("  Calculando streak_actual...")
    stg_sorted_pid = stg_sorted.sort_values(["pledge_id", "attempt_ts"])
    # is_failure flag
    stg_sorted_pid["is_failure"] = (1 - stg_sorted_pid["is_success"]).astype("int8")
    # corremos un cumcount de fallos consecutivos por pledge (igual al original pero correcto)
    is_succ = stg_sorted_pid["is_success"].values
    pid_arr = stg_sorted_pid["pledge_id"].values
    # streak = #fallos al final desde el último éxito (o desde el inicio)
    # Calculamos un "id de bloque" que cambia con cada éxito
    block_id = np.cumsum(is_succ)  # se incrementa con cada éxito
    # combinar con pledge para que los bloques no crucen pledges:
    pid_codes = pd.Categorical(pid_arr).codes
    composite_block = pid_codes.astype("int64") * (block_id.max() + 1) + block_id
    # streak position: cumcount dentro del bloque, contando solo fallos
    streak_position = pd.Series(stg_sorted_pid["is_failure"].values).groupby(composite_block).cumsum().values
    stg_sorted_pid["streak_running"] = streak_position
    # streak en cada momento es la posición de fallo dentro del bloque (0 si fue éxito)
    # NOTA: queremos streak HASTA y SIN INCLUIR el snapshot. Tomamos el último valor < snapshot via merge_asof.
    asof_streak = pd.merge_asof(
        snap_sorted[["pledge_id", "snapshot_ts"]].sort_values("snapshot_ts"),
        stg_sorted_pid[["pledge_id", "attempt_ts", "streak_running"]].sort_values("attempt_ts"),
        left_on="snapshot_ts", right_on="attempt_ts", by="pledge_id",
        direction="backward", allow_exact_matches=False
    )
    asof_streak["streak_actual"] = asof_streak["streak_running"].fillna(0).astype("int32")

    # Combinar todo
    print("[5/5] Ensamblando features finales...")
    feat = snap_sorted.copy()
    feat["snap_month"] = pd.to_datetime(feat["snapshot_ts"].values.astype("datetime64[M]")).astype("datetime64[ns]")

    # Normalizar tipos de las tablas auxiliares para merges seguros
    cum["snap_month"] = pd.to_datetime(cum["snap_month"].values.astype("datetime64[M]")).astype("datetime64[ns]")
    last_m["snap_month"] = pd.to_datetime(last_m["snap_month"].values.astype("datetime64[M]")).astype("datetime64[ns]")

    feat = feat.merge(asof_last[["pledge_id", "snapshot_ts", "days_since_last_attempt", "last_attempt_success", "last_payment_amount", "last_payment_method", "last_bank_name", "last_gateway"]],
                       on=["pledge_id", "snapshot_ts"], how="left")
    feat = feat.merge(cum[["pledge_id", "snap_month", "cum_intentos", "cum_exitos", "cum_fallos", "success_rate_hist"]],
                       on=["pledge_id", "snap_month"], how="left")
    feat = feat.merge(last_m, on=["pledge_id", "snap_month"], how="left")
    feat[["n_intentos_30d", "n_exitos_30d", "n_fallos_30d"]] = feat[["n_intentos_30d", "n_exitos_30d", "n_fallos_30d"]].fillna(0).astype("int32")
    feat = feat.merge(asof_streak[["pledge_id", "snapshot_ts", "streak_actual"]],
                       on=["pledge_id", "snapshot_ts"], how="left")
    feat["streak_actual"] = feat["streak_actual"].fillna(0).astype("int32")
    feat["tenure_days"] = (feat["snapshot_ts"] - feat["effective_start"]).dt.days.astype("int32")

    feat = feat.drop(columns=["effective_start", "npsp_end_date_c", "snap_month"], errors="ignore")

    # === CAMBIO (v9 — Motor 2 a nivel donante): resolver y adjuntar contact_id ===
    # Un pledge DEBERÍA tener un único contact_id en todo su historial de
    # intentos. Si el CRM tiene algún caso inconsistente (más de un contact_id
    # para el mismo pledge_id), se resuelve tomando el más frecuente en su
    # historial — misma regla que usa 01b_motor2_dataset_donante.py, para que
    # ambos scripts sean consistentes entre sí si en algún punto se fusionan.
    print("  Resolviendo contact_id por pledge...")
    conteo_contacto = (stg.groupby(["pledge_id", "contact_id"]).size()
                        .reset_index(name="n_intentos_con_ese_contacto"))
    n_pledges_ambiguos = int((conteo_contacto.groupby("pledge_id")["contact_id"].nunique() > 1).sum())
    if n_pledges_ambiguos:
        print(f"  [AVISO] {n_pledges_ambiguos:,} pledges con más de un contact_id "
              f"distinto en su historial — resuelto tomando el más frecuente.")
    pledge_to_contact = (conteo_contacto
                          .sort_values("n_intentos_con_ese_contacto", ascending=False)
                          .drop_duplicates(subset=["pledge_id"], keep="first")
                          [["pledge_id", "contact_id"]])
    feat = feat.merge(pledge_to_contact, on="pledge_id", how="left")
    sin_contacto = int(feat["contact_id"].isna().sum())
    if sin_contacto:
        print(f"  [AVISO] {sin_contacto:,} filas ({100*sin_contacto/len(feat):.2f}%) "
              f"sin contact_id resuelto (pledge sin intentos en el staging).")

    out_path = OUT_PATH
    feat.to_parquet(out_path, index=False)
    print(f"  filas finales: {len(feat):,}, cols: {feat.columns.tolist()}")
    print(f"  Tasa churn 30d global: {feat['label_churn_30d'].mean():.4f}")
    print(f"  Por año:")
    print(feat.groupby(feat['snapshot_ts'].dt.year).agg(n=("label_churn_30d", "size"), tasa=("label_churn_30d", "mean")))
    print(f"\n[OK] {out_path}")


if __name__ == "__main__":
    main()
