#!/usr/bin/env bash
# =============================================================================
# run_all.sh — pipeline completo UNICEF Colombia (v8, estructura plana)
#
# Corre el pipeline end-to-end desde los CSV crudos hasta la tabla maestra de
# donantes: preprocesamiento → features de los 3 motores → entrenamiento de
# Motor 1 y Motor 2 → tabla intermedia P1+P2. Mide tiempos y valida el número
# de filas/columnas de cada parquet contra los valores esperados.
#
# ESTRUCTURA PLANA (v8):
#   v8_entregable/
#   ├── raw/          ← CSVs crudos del cliente (9 archivos)
#   ├── data/         ← unified, staging, features, predictions, entregable
#   ├── results/      ← modelos y métricas de Motor 1 y Motor 2
#   └── scripts/      ← este pipeline
#
# IDEMPOTENCIA:
#   Cada etapa se SALTA si su parquet de salida ya existe. Esto hace el
#   pipeline resumible y evita re-generar artefactos pesados que no cambiaron.
#   Para forzar el re-cálculo de una etapa, borra su salida (o corre con
#   FORCE=1 para re-generar TODO desde cero).
#
# Uso:
#     source /ruta/al/proyecto/.venv/bin/activate     # venv con las deps
#     bash v8_entregable/scripts/run_all.sh
#     FORCE=1 bash v8_entregable/scripts/run_all.sh    # re-genera todo
#
#     # Con otra fecha de corte — FORCE=1 es OBLIGATORIO (ver IDEMPOTENCIA arriba):
#     FORCE=1 CUTOFF_DATE=2026-12-31 bash v8_entregable/scripts/run_all.sh
#
# ALCANCE:
#   Este script es el RE-ENTRENAMIENTO (anual o semestral). NO incluye la
#   puntuación mensual de Motor 2 (03_train/04_score_motor2.py), que responde
#   "quién cancela este mes" y se corre cada mes contra el modelo ya entrenado.
#   Ver scripts/ORDEN_DE_EJECUCION.md → "Entrenar vs. puntuar (Motor 2)".
#
# Compatible con bash 3.2+ (macOS default) y bash 4+.
# =============================================================================
set -uo pipefail

# ---------- Configuración ----------
# ROOT = raíz del entregable (carpeta que contiene scripts/, data/, raw/).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$ROOT/scripts"
LOG="$ROOT/run_all.log"
FORCE="${FORCE:-0}"

# Fecha de corte del pipeline. Cambia con la variable de entorno CUTOFF_DATE.
# Ejemplos:
#   bash scripts/run_all.sh                          → usa 2025-12-31 (default)
#   CUTOFF_DATE=2026-03-31 bash scripts/run_all.sh   → corte en marzo 2026
CUTOFF_DATE="${CUTOFF_DATE:-2025-12-31}"
CUTOFF_ARG="--cutoff-date $CUTOFF_DATE"
# Slug para nombres de archivo de salida (2025-12-31 → 2025_12_31)
CUTOFF_SLUG="${CUTOFF_DATE//-/_}"

# Fecha de ejecución ("hoy"): desde aquí la etapa de timing proyecta las fechas
# concretas de próximo cobro. Se pasa EXPLÍCITAMENTE para que quede en el log; si
# no, el script usaría la fecha del sistema sin dejar rastro de cuál fue.
# El timing operativo se refresca cada mes con run_monthly.sh; el que sale de acá
# es el del momento del re-entrenamiento.
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
RUN_DATE_ARG="--run-date $RUN_DATE"

# ANSI colors
G='\033[1;32m'; R='\033[1;31m'; Y='\033[1;33m'; B='\033[1;34m'; N='\033[0m'

: > "$LOG"

# Estado global: listas paralelas (compatibles bash 3.2)
STEP_LABELS=(); STEP_RESULTS=(); STEP_TIMES=()
record() { STEP_LABELS+=("$1"); STEP_RESULTS+=("$2"); STEP_TIMES+=("$3"); }

# ---------- Helpers ----------
say() { printf '%b\n' "$*" | tee -a "$LOG" ; }
hr()  { say "${B}--------------------------------------------------------------------${N}" ; }

verify_parquet() {
    local path="$1" exp_rows="$2" exp_cols="$3"
    if [ ! -f "$path" ]; then
        say "${R}  archivo no existe: $path${N}"
        return 1
    fi
    python3 - "$path" "$exp_rows" "$exp_cols" <<'PY'
import sys, pyarrow.parquet as pq
path, exp_rows, exp_cols = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = pq.read_metadata(path)
rows, cols = m.num_rows, m.num_columns
ok = (rows == exp_rows) and (cols == exp_cols)
print(f"  filas: {rows:>10,} (esperado {exp_rows:,})   cols: {cols} (esperado {exp_cols})", end="")
print("   OK" if ok else "   DIFF")
sys.exit(0 if ok else 1)
PY
}

# Ejecuta un comando midiendo tiempo. $1=etiqueta  $2=comando
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

# Ejecuta una etapa SOLO si su parquet de salida no existe (idempotencia).
# $1=etiqueta  $2=comando  $3=parquet de salida a chequear
run_or_skip() {
    local label="$1" cmd="$2" out="$3"
    if [ "$FORCE" != "1" ] && [ -f "$out" ]; then
        say ""; say "${B}[SKIP] ${label} — salida ya existe ($(basename "$out"))${N}"
        record "$label" "SKIP" "-"; return 0
    fi
    run_step "$label" "$cmd"
}

verify_step() {
    local label="$1" path="$2" rows="$3" cols="$4"
    say ""; say "${B}[VERIFICAR] ${label}${N}"
    if verify_parquet "$path" "$rows" "$cols" | tee -a "$LOG" ; then
        record "VERIFY: $label" "PASS" "-"; return 0
    else
        record "VERIFY: $label" "FAIL" "-"; return 1
    fi
}

# ---------- Comprobaciones previas ----------
say "${B}=====================================================================${N}"
say "${B}  PIPELINE COMPLETO — UNICEF Colombia (v8)${N}"
say "${B}  $(date)${N}"
say "${B}  ROOT        = $ROOT${N}"
say "${B}  CUTOFF_DATE = $CUTOFF_DATE  (frontera de información: ventanas y splits)${N}"
say "${B}  RUN_DATE    = $RUN_DATE  (hoy: desde aquí se proyectan las fechas de cobro)${N}"
say "${B}=====================================================================${N}"

# Aviso de idempotencia: las etapas se SALTAN si su salida ya existe. Al cambiar
# la fecha de corte eso dejaría el dataset de Motor 2 y los modelos del corte
# anterior. 02_train_motor2.py aborta si detecta la inconsistencia, pero es mejor
# advertirlo acá antes de gastar tiempo.
if [ "$CUTOFF_DATE" != "2025-12-31" ] && [ "$FORCE" != "1" ]; then
    say ""
    say "${Y}[AVISO] Corte distinto al default y FORCE != 1.${N}"
    say "${Y}        Las etapas cuya salida ya existe se van a SALTAR y quedarían${N}"
    say "${Y}        artefactos del corte anterior. Para re-generar todo:${N}"
    say "${Y}          FORCE=1 CUTOFF_DATE=$CUTOFF_DATE bash scripts/run_all.sh${N}"
fi

if [ ! -d "$ROOT/raw" ]; then
    say "${R}ERROR: no encuentro $ROOT/raw (CSVs crudos del cliente).${N}"; exit 1
fi
if [ ! -d "$SCRIPTS" ]; then
    say "${R}ERROR: no encuentro $SCRIPTS${N}"; exit 1
fi
python3 -c "import pandas, pyarrow, numpy" 2>/dev/null || {
    say "${R}ERROR: faltan dependencias Python. Instala con:${N}"
    say "${Y}  pip install -r $ROOT/requirements.txt${N}"; exit 1
}

# ---------- Etapa 0: Preprocesamiento ----------
run_or_skip "00 · unify_raw_data" \
    "python3 \"$SCRIPTS/00_preprocesamiento/00_unify_raw_data.py\" --raw-dir \"$ROOT\" --out-dir \"$ROOT/data/unified\"" \
    "$ROOT/data/unified/payments/payments_unified.parquet" || exit 1
verify_step "contact_unified"      "$ROOT/data/unified/contact/contact_unified.parquet"          503098  158
verify_step "opportunity_unified"  "$ROOT/data/unified/opportunity/opportunity_unified.parquet"  2917121 72
verify_step "pledge_unified"       "$ROOT/data/unified/pledge/pledge_unified.parquet"            464807  117
verify_step "payments_unified"     "$ROOT/data/unified/payments/payments_unified.parquet"        5566032 36
verify_step "allocation_unified"   "$ROOT/data/unified/allocation/allocation_unified.parquet"    276004  67

run_or_skip "01 · create_staging" \
    "python3 \"$SCRIPTS/00_preprocesamiento/01_create_staging.py\"" \
    "$ROOT/data/staging/stg_payment_attempts.parquet" || exit 1
verify_step "stg_payment_attempts" "$ROOT/data/staging/stg_payment_attempts.parquet" 5309073 17

# ---------- Etapa 1: Features Motor 1 ----------
run_or_skip "M1 · clean_bank_normalization" \
    "python3 \"$SCRIPTS/01_features_motor1/01_clean_bank_normalization.py\"" \
    "$ROOT/data/features/bank_canonical.parquet" || exit 1
verify_step "bank_canonical" "$ROOT/data/features/bank_canonical.parquet" 5309073 1

run_or_skip "M1 · enrich_with_contact" \
    "python3 \"$SCRIPTS/01_features_motor1/02_enrich_with_contact.py\"" \
    "$ROOT/data/features/motor1_features.parquet" || exit 1
verify_step "motor1_features" "$ROOT/data/features/motor1_features.parquet" 5309073 20

run_or_skip "M1 · build_allocation_features" \
    "python3 \"$SCRIPTS/01_features_motor1/03_build_allocation_features.py\"" \
    "$ROOT/data/features/allocation_features.parquet" || exit 1
verify_step "allocation_features" "$ROOT/data/features/allocation_features.parquet" 5309073 15

# ---------- Etapa 2: Features Motor 2 ----------
run_or_skip "M2 · motor2_dataset" \
    "python3 \"$SCRIPTS/02_features_motor2/01_motor2_dataset.py\" $CUTOFF_ARG" \
    "$ROOT/data/features/motor2_pledge_month.parquet" || exit 1
# El número de filas depende del corte (la ventana de 24 snapshots se mueve),
# así que solo se verifica contra el valor esperado en el corte por defecto.
# CAMBIO (Motor 2 a nivel donante): 18 → 19 columnas. 01_motor2_dataset.py ahora
# adjunta contact_id, para que 01b pueda agrupar pledges del mismo donante sin
# releer el staging.
if [ "$CUTOFF_DATE" = "2025-12-31" ]; then
    verify_step "motor2_pledge_month" "$ROOT/data/features/motor2_pledge_month.parquet" 2548075 19
else
    say ""; say "${Y}[VERIFICAR] motor2_pledge_month — omitido (corte no default: $CUTOFF_DATE)${N}"
fi

# ---------- Etapa 2.5: Entrenamientos ----------
run_or_skip "M1 · train_motor1" \
    "python3 \"$SCRIPTS/03_train/01_train_motor1.py\" $CUTOFF_ARG" \
    "$ROOT/results/motor1/test_predictions_motor1.parquet" || exit 1

run_or_skip "M2 · train_motor2" \
    "python3 \"$SCRIPTS/03_train/02_train_motor2.py\" $CUTOFF_ARG" \
    "$ROOT/results/motor2/test_predictions_motor2.parquet" || exit 1

# ---------- Etapa 2.5b: Motor 2 a nivel DONANTE ----------
# Corre EN PARALELO al pledge-level, no lo reemplaza. La unidad de predicción
# pasa de (pledge, snapshot) a (donante, snapshot): un donante abandona solo si
# cancela TODOS sus pledges activos dentro de la misma ventana de 30 días.
# Ver docs/Cambios_Motor_2.pdf para la justificación y las reglas de agregación.
run_or_skip "M2d · motor2_dataset_donante" \
    "python3 \"$SCRIPTS/02_features_motor2/01b_motor2_dataset_donante.py\"" \
    "$ROOT/data/features/motor2_donante_month.parquet" || exit 1
say "  Dataset donante: data/features/motor2_donante_month.parquet"

run_or_skip "M2d · train_motor2_donante" \
    "python3 \"$SCRIPTS/03_train/02b_train_motor2_donante.py\" $CUTOFF_ARG" \
    "$ROOT/results/motor2_donante/test_predictions_motor2_donante.parquet" || exit 1

# ---------- Etapa 2.6: Timing de reintento Motor 1 ----------
run_or_skip "M1 · timing_reintento" \
    "python3 \"$SCRIPTS/03_train/03_timing_reintento_motor1.py\" $CUTOFF_ARG $RUN_DATE_ARG" \
    "$ROOT/results/motor1/timing_reintento_motor1.parquet" || exit 1
say "  Timing optimizer: $(basename $ROOT/results/motor1/timing_reintento_motor1.csv)"

# ---------- Etapa 3: Tabla intermedia Motor 3 ----------
TABLA_OUT="$ROOT/data/entregable/Tabla_Donantes_P1_P2_${CUTOFF_SLUG}.parquet"
run_or_skip "M3 · tabla_donantes_p1_p2" \
    "python3 \"$SCRIPTS/04_features_motor3/01_tabla_donantes_p1_p2.py\" $CUTOFF_ARG" \
    "$TABLA_OUT" || exit 1
# Nota: no se verifica row count porque varía según la fecha de corte.
say "  Tabla generada: $(basename $TABLA_OUT)"

# ---------- Resumen final ----------
hr; say ""
say "${B}=====================================================================${N}"
say "${B}  RESUMEN${N}"
say "${B}=====================================================================${N}"
PASS=0 ; FAIL=0 ; SKIP=0
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
say "  Log completo: $LOG"
hr

[ "$FAIL" -gt 0 ] && exit 1
exit 0
