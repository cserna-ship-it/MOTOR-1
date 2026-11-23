#!/usr/bin/env bash
# =============================================================================
# run_monthly.sh — Ejecución mensual de inferencia UNICEF Colombia
#
# PROPÓSITO
# ---------
# Genera las tres salidas operativas que el equipo consume cada mes:
#
#   1. timing_reintento_motor1.csv   → fecha óptima del próximo cobro por donante
#   2. scoring_motor2_<hoy>.csv      → riesgo de cancelación del MES EN CURSO
#                                      (lista de llamadas de retención)
#   3. Tabla_Donantes_P1_P2_*.csv    → tabla maestra con P1 (éxito cobro) y
#                                      P2 (riesgo de abandono) por donante
#
# NO RE-ENTRENA ningún modelo: las tres etapas cargan los modelos ya entrenados.
# Para re-entrenar usa run_all.sh (proceso anual/semestral).
#
# LAS DOS FECHAS:
#   RUN_DATE    = hoy. Define desde dónde se proyectan las fechas de cobro y cuál
#                 es el snapshot de Motor 2 (el día 1 de ese mes).
#   CUTOFF_DATE = corte con que se entrenó el modelo. Solo nombra la tabla maestra
#                 y define el año de agregación de P1. NO se le pasa a la etapa de
#                 timing (ver nota en esa etapa).
#
# DOS CAMINOS DE DATOS (por eso hay dos insumos):
#   - Motor 1 usa el archivo de pagos YTD del año en curso (PAYMENTS_YTD), que se
#     encadena sobre las features históricas sin re-procesar 5,3M de filas.
#   - Motor 2 arma su snapshot desde el STAGING, porque necesita `bank_name`, que
#     el archivo de features YTD no trae. Si al staging le faltan los pagos del
#     mes que cerró, este script lo regenera desde raw/ (etapa C, ~12 min).
#
# PRE-REQUISITOS
# --------------
# Los siguientes archivos deben existir antes de correr este script:
#
#   results/motor1/motor1_catboost.cbm     ← modelo Motor 1 entrenado
#   results/motor1/motor1_calibrador.pkl   ← calibrador isotónico
#   results/motor2/motor2_catboost.cbm     ← modelo Motor 2 entrenado
#   results/motor2/motor2_calibrador.pkl   ← calibrador isotónico de Motor 2
#   data/features/motor1_features.parquet  ← features históricas (generadas por run_all.sh)
#   data/features/allocation_features.parquet
#   data/features/motor2_pledge_month.parquet
#   results/motor1/test_predictions_motor1.parquet
#   results/motor2/test_predictions_motor2.parquet
#   data/unified/pledge/pledge_unified.parquet
#
# Si alguno de estos archivos no existe, corre primero run_all.sh.
#
# PARÁMETROS
# ----------
# PAYMENTS_YTD → ruta al archivo de pagos del año en curso (CSV o parquet).
#                El equipo lo exporta del CRM y lo REEMPLAZA cada mes.
#                default: raw/payments_ytd.parquet
# CUTOFF_DATE  → fecha de corte del modelo entrenado (NO cambiar hasta re-entrenar)
#                default: 2025-12-31
# RUN_DATE     → "hoy": la fecha desde la que se proyectan los reintentos
#                default: fecha actual del sistema
#
# EJEMPLOS DE USO
# ---------------
#   # Ejecución normal (usa fecha de hoy como RUN_DATE)
#   PAYMENTS_YTD=raw/payments_2026_ytd.csv bash scripts/run_monthly.sh
#
#   # Con todos los parámetros explícitos
#   PAYMENTS_YTD=raw/payments_2026_ytd.csv \
#   CUTOFF_DATE=2025-12-31 \
#   RUN_DATE=2026-07-31 \
#   bash scripts/run_monthly.sh
#
# Compatible con bash 3.2+ (macOS default) y bash 4+.
# =============================================================================
set -uo pipefail

# ---------- Configuración ----------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
LOG="$ROOT/run_monthly.log"

# Fecha de corte del modelo (fija hasta el próximo re-entrenamiento)
CUTOFF_DATE="${CUTOFF_DATE:-2025-12-31}"
CUTOFF_ARG="--cutoff-date $CUTOFF_DATE"
CUTOFF_SLUG="${CUTOFF_DATE//-/_}"

# Fecha de ejecución ("hoy" — proyecta fechas de reintento desde este punto)
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
RUN_DATE_ARG="--run-date $RUN_DATE"

# Snapshot de Motor 2 a puntuar: el primer día del mes de RUN_DATE.
# (2026-02-05 → 2026-02-01). Sus features usan solo intentos anteriores a esa
# fecha, así que conviene correr en los primeros días del mes.
SNAPSHOT_DATE="${RUN_DATE%-*}-01"
SNAPSHOT_SLUG="${SNAPSHOT_DATE//-/_}"
M2_SNAPSHOT="$ROOT/data/features/motor2_snapshot_${SNAPSHOT_SLUG}.parquet"
M2_SCORING="$ROOT/results/motor2/scoring_motor2_${RUN_DATE//-/_}.parquet"

# Archivo de pagos YTD (el equipo lo reemplaza mensualmente)
PAYMENTS_YTD="${PAYMENTS_YTD:-$ROOT/raw/payments_ytd.parquet}"
# Detectar si existe versión .csv como alternativa
if [ ! -f "$PAYMENTS_YTD" ] && [ -f "${PAYMENTS_YTD%.parquet}.csv" ]; then
    PAYMENTS_YTD="${PAYMENTS_YTD%.parquet}.csv"
fi
YTD_FEATURES="$ROOT/data/features/motor1_features_ytd.parquet"
STAGING="$ROOT/data/staging/stg_payment_attempts.parquet"

# Refresco del staging: Motor 2 construye su snapshot desde el staging, no desde
# el archivo YTD (que no trae bank_name). Si el staging no llega hasta el mes
# anterior al snapshot, hay que regenerarlo desde raw/.
#   REFRESH_STAGING=auto (default) → se refresca solo si le faltan datos
#   REFRESH_STAGING=1              → se refresca siempre
#   REFRESH_STAGING=0              → nunca (útil si ya se corrió aparte)
REFRESH_STAGING="${REFRESH_STAGING:-auto}"

# ANSI colors
G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; B='\033[1;34m'; N='\033[0m'

: > "$LOG"

STEP_LABELS=(); STEP_RESULTS=(); STEP_TIMES=()
record() { STEP_LABELS+=("$1"); STEP_RESULTS+=("$2"); STEP_TIMES+=("$3"); }

say() { printf '%b\n' "$*" | tee -a "$LOG" ; }
hr()  { say "${B}--------------------------------------------------------------------${N}" ; }

run_step() {
    local label="$1" cmd="$2"
    say ""; hr
    say "${B}[ETAPA] ${label}${N}"
    say "${Y}> ${cmd}${N}"; hr
    local t0=$(date +%s)
    if eval "$cmd" 2>&1 | tee -a "$LOG" ; then
        local t1=$(date +%s); local secs=$((t1-t0))
        record "$label" "PASS" "${secs}s"
        say "${G}[OK] ${label}  (${secs}s)${N}"; return 0
    else
        record "$label" "FAIL" "-"
        say "${R}[ERROR] ${label} falló${N}"; return 1
    fi
}

# ---------- Header ----------
say "${B}=====================================================================${N}"
say "${B}  EJECUCIÓN MENSUAL — UNICEF Colombia${N}"
say "${B}  $(date)${N}"
say "${B}  CUTOFF_DATE   = $CUTOFF_DATE  (modelo entrenado — solo nombra la tabla y el año de P1)${N}"
say "${B}  RUN_DATE      = $RUN_DATE  (hoy — proyecta fechas de cobro y define el snapshot)${N}"
say "${B}  SNAPSHOT M2   = $SNAPSHOT_DATE  (riesgo de cancelación de este mes)${N}"
say "${B}  PAYMENTS_YTD  = $PAYMENTS_YTD${N}"
say "${B}=====================================================================${N}"

# ---------- Verificar pre-requisitos ----------
say ""
say "${B}[VERIFICANDO PRE-REQUISITOS]${N}"

MISSING=0
check_file() {
    local f="$1" label="$2"
    if [ -f "$f" ]; then
        say "  ${G}OK${N}  $label"
    else
        say "  ${R}FALTA${N}  $label  →  $f"
        MISSING=$((MISSING+1))
    fi
}

check_file "$ROOT/results/motor1/motor1_catboost.cbm"             "Motor 1 — modelo"
check_file "$ROOT/results/motor1/motor1_calibrador.pkl"           "Motor 1 — calibrador"
check_file "$ROOT/results/motor2/motor2_catboost.cbm"             "Motor 2 — modelo"
check_file "$ROOT/results/motor2/motor2_calibrador.pkl"           "Motor 2 — calibrador"
check_file "$ROOT/data/features/motor1_features.parquet"          "Features Motor 1 (histórico)"
check_file "$ROOT/data/features/allocation_features.parquet"      "Allocation features"
check_file "$ROOT/data/features/motor2_pledge_month.parquet"      "Features Motor 2"
check_file "$ROOT/results/motor1/test_predictions_motor1.parquet" "Predicciones Motor 1"
check_file "$ROOT/results/motor2/test_predictions_motor2.parquet" "Predicciones Motor 2"
check_file "$ROOT/data/unified/pledge/pledge_unified.parquet"     "Pledge unified"
check_file "$ROOT/data/unified/opportunity/opportunity_unified.parquet" "Opportunity unified"
check_file "$PAYMENTS_YTD"                                         "Pagos YTD (insumo mensual)"

if [ "$MISSING" -gt 0 ]; then
    say ""
    say "${R}ERROR: Faltan $MISSING archivo(s).${N}"
    if [ ! -f "$PAYMENTS_YTD" ]; then
        say "${Y}  → Coloca el archivo de pagos YTD en: $PAYMENTS_YTD${N}"
        say "${Y}    (o especifica la ruta con: PAYMENTS_YTD=/ruta/al/archivo.csv bash scripts/run_monthly.sh)${N}"
    fi
    say "${Y}  → Para los demás archivos, corre primero: bash scripts/run_all.sh${N}"
    exit 1
fi
say ""

# ---------- Etapa A: Features incrementales (año en curso) ----------
run_step "Features incrementales YTD" \
    "python3 \"$SCRIPTS/01_features_motor1/04_build_incremental_features.py\" \
        --payments-ytd \"$PAYMENTS_YTD\"" || exit 1

if [ -f "$YTD_FEATURES" ]; then
    NROWS=$(python3 -c "import pyarrow.parquet as pq; m = pq.read_metadata('$YTD_FEATURES'); print(f'{m.num_rows:,}')" 2>/dev/null || echo "?")
    say "  → motor1_features_ytd.parquet  ($NROWS intentos del año en curso)"
fi

# ---------- Etapa B: Timing óptimo de reintento ----------
# El corte que recibe este script NO es el del modelo, es la frontera de
# información: hasta cuándo se conocen los intentos. En una corrida mensual eso
# es HOY (RUN_DATE), no la fecha de entrenamiento. Con el corte del modelo la
# ventana móvil de 12 meses se quedaría anclada en el año del entrenamiento e
# ignoraría los intentos del año en curso — justamente los del archivo YTD.
run_step "Timing óptimo de reintento (Motor 1)" \
    "python3 \"$SCRIPTS/03_train/03_timing_reintento_motor1.py\" \
        --cutoff-date $RUN_DATE $RUN_DATE_ARG \
        --features-file \"$YTD_FEATURES\"" || exit 1

TIMING_CSV="$ROOT/results/motor1/timing_reintento_motor1.csv"
if [ -f "$TIMING_CSV" ]; then
    NROWS=$(python3 -c "import pyarrow.parquet as pq; m = pq.read_metadata('${TIMING_CSV/.csv/.parquet}'); print(f'{m.num_rows:,}')" 2>/dev/null || echo "?")
    say "  → ${TIMING_CSV##*/}   ($NROWS donantes activos, último intento exitoso o fallido)"
fi

# ---------- Etapa C: Staging al día (pre-requisito de Motor 2) ----------
# Motor 2 arma su snapshot desde el staging, no desde el archivo YTD (que no trae
# bank_name, una de las features del modelo). Para puntuar el snapshot del día 1
# de este mes, el staging tiene que llegar al menos hasta el último día del mes
# anterior.
MES_ANTERIOR_FIN=$(python3 -c "
import pandas as pd, sys
print((pd.Timestamp('$SNAPSHOT_DATE') - pd.Timedelta(days=1)).date())" 2>/dev/null)

staging_al_dia() {
    python3 - "$STAGING" "$MES_ANTERIOR_FIN" <<'PY' 2>/dev/null
import sys, pandas as pd
try:
    mx = pd.read_parquet(sys.argv[1], columns=["attempt_ts"])["attempt_ts"].max()
except Exception:
    sys.exit(1)
sys.exit(0 if mx >= pd.Timestamp(sys.argv[2]) else 1)
PY
}

NECESITA_REFRESH=0
case "$REFRESH_STAGING" in
    1)    NECESITA_REFRESH=1 ;;
    0)    NECESITA_REFRESH=0 ;;
    auto) staging_al_dia || NECESITA_REFRESH=1 ;;
esac

if [ "$NECESITA_REFRESH" = "1" ]; then
    say ""
    say "${Y}[STAGING] Al staging le faltan datos hasta $MES_ANTERIOR_FIN — se regenera desde raw/ (~12 min).${N}"
    say "${Y}          Los pagos del mes nuevo deben estar en raw/ Y registrados en FILE_GROUPS${N}"
    say "${Y}          (scripts/00_preprocesamiento/00_unify_raw_data.py).${N}"
    run_step "Unificación de CSVs crudos" \
        "python3 \"$SCRIPTS/00_preprocesamiento/00_unify_raw_data.py\" --raw-dir \"$ROOT\" --out-dir \"$ROOT/data/unified\"" || exit 1
    run_step "Staging de intentos de cobro" \
        "python3 \"$SCRIPTS/00_preprocesamiento/01_create_staging.py\"" || exit 1
    if ! staging_al_dia; then
        say ""
        say "${R}ERROR: incluso después de regenerar, el staging no llega a $MES_ANTERIOR_FIN.${N}"
        say "${Y}  → Los pagos del mes nuevo no están registrados en FILE_GROUPS.${N}"
        say "${Y}    Agrega el archivo a la lista 'payments' en 00_unify_raw_data.py y reintenta.${N}"
        exit 1
    fi
else
    say ""
    say "${G}[STAGING] Ya cubre hasta $MES_ANTERIOR_FIN — no se regenera.${N}"
    record "Staging al día (sin refresco)" "SKIP" "-"
fi

# ---------- Etapa D: Snapshot del mes para Motor 2 ----------
run_step "Snapshot Motor 2 ($SNAPSHOT_DATE)" \
    "python3 \"$SCRIPTS/02_features_motor2/01_motor2_dataset.py\" \
        --cutoff-date $RUN_DATE --months 1 --out \"$M2_SNAPSHOT\"" || exit 1

# ---------- Etapa E: Puntuación de Motor 2 (lista de cancelaciones) ----------
run_step "Puntuación Motor 2 — riesgo de cancelación del mes" \
    "python3 \"$SCRIPTS/03_train/04_score_motor2.py\" \
        --score-date $RUN_DATE --dataset \"$M2_SNAPSHOT\"" || exit 1

M2_SCORING_CSV="${M2_SCORING/.parquet/.csv}"
if [ -f "$M2_SCORING_CSV" ]; then
    say "  → ${M2_SCORING_CSV##*/}   (lista de retención, ordenada por prob_churn)"
fi

# ---------- Etapa E2: Puntuación Motor 2 a nivel DONANTE ----------
# Misma idea que la etapa E pero la unidad es el donante, no el pledge: un
# donante en riesgo es el que va a cancelar TODOS sus compromisos. Se salta (sin
# fallar) si el modelo de donante no está entrenado — es una salida adicional,
# no un pre-requisito de la tabla maestra.
M2D_MODEL="$ROOT/results/motor2_donante/motor2_donante_catboost.cbm"
if [ -f "$M2D_MODEL" ]; then
    run_step "Puntuación Motor 2 DONANTE — riesgo de abandono del mes" \
        "python3 \"$SCRIPTS/03_train/04b_score_motor2_donante.py\" \
            --score-date $RUN_DATE --dataset \"$M2_SNAPSHOT\"" || exit 1
    say "  → scoring_motor2_donante_${RUN_DATE//-/_}.csv   (lista de retención a nivel donante)"
else
    say ""
    say "${Y}[SKIP] Motor 2 donante — no existe $M2D_MODEL.${N}"
    say "${Y}       Entrénalo con run_all.sh (etapa M2d) si quieres esta salida.${N}"
    record "Puntuación Motor 2 DONANTE" "SKIP" "-"
fi

# ---------- Etapa F: Tabla Donantes P1 + P2 ----------
# Se le pasa el scoring del mes para que la columna P2 sea la del mes EN CURSO y
# no la del snapshot con que se entrenó el modelo.
TABLA_OUT="$ROOT/data/entregable/Tabla_Donantes_P1_P2_${CUTOFF_SLUG}.parquet"
run_step "Tabla Donantes P1 + P2" \
    "python3 \"$SCRIPTS/04_features_motor3/01_tabla_donantes_p1_p2.py\" $CUTOFF_ARG \
        --p2-scoring \"$M2_SCORING\"" || exit 1

TABLA_CSV="${TABLA_OUT/.parquet/.csv}"
if [ -f "$TABLA_CSV" ]; then
    NROWS=$(python3 -c "import pyarrow.parquet as pq; m = pq.read_metadata('$TABLA_OUT'); print(f'{m.num_rows:,}')" 2>/dev/null || echo "?")
    say "  → ${TABLA_CSV##*/}   ($NROWS donantes)"
fi

# ---------- Resumen final ----------
hr; say ""
say "${B}=====================================================================${N}"
say "${B}  RESUMEN${N}"
say "${B}=====================================================================${N}"
PASS=0; FAIL=0; SKIP=0
n=${#STEP_LABELS[@]}; i=0
while [ $i -lt $n ]; do
    lbl="${STEP_LABELS[$i]}"; res="${STEP_RESULTS[$i]}"; tim="${STEP_TIMES[$i]}"
    case "$res" in
        PASS) say "  ${G}PASS${N}   ${lbl}   (${tim})"; PASS=$((PASS+1)) ;;
        FAIL) say "  ${R}FAIL${N}   ${lbl}";            FAIL=$((FAIL+1)) ;;
        SKIP) say "  ${Y}SKIP${N}   ${lbl}";            SKIP=$((SKIP+1)) ;;
    esac
    i=$((i+1))
done
say ""
say "  Total: ${G}${PASS} PASS${N}  ${R}${FAIL} FAIL${N}  ${Y}${SKIP} SKIP${N}"
say ""
say "${B}  ARCHIVOS GENERADOS:${N}"
say "  data/features/motor1_features_ytd.parquet     (features del año en curso)"
say "  results/motor1/timing_reintento_motor1.csv   (fechas óptimas de cobro)"
say "  results/motor2/scoring_motor2_${RUN_DATE//-/_}.csv  (riesgo de cancelación del mes)"
say "  data/entregable/Tabla_Donantes_P1_P2_${CUTOFF_SLUG}.csv  (tabla maestra donantes)"
say ""
say "  Log completo: $LOG"
hr

[ "$FAIL" -gt 0 ] && exit 1
exit 0
