#!/usr/bin/env python3
"""
01_create_staging.py — ETAPA 0 del pipeline (preparación de la tabla base)

PROPÓSITO
---------
Convertir las 5 tablas unificadas (payments, opportunity, pledge, contact,
allocation) en UNA tabla canónica de "intentos de cobro": `stg_payment_attempts`.
Es la base sobre la que TODOS los motores construyen sus features.

ESTRUCTURA DEL JOIN
-------------------
El modelo relacional de Salesforce NPSP:

    PAYMENT ──[crm_donation_id]──► OPPORTUNITY ──[npe03_recurring_donation_c]──► PLEDGE
    (intento)                      (donación)                                    (contrato)

- Payment = un intento individual de cobro (exitoso o fallido).
- Opportunity = la donación "lógica" de Salesforce. Puente entre Payment y Pledge.
- Pledge = el contrato de donación recurrente (mensual, etc.).

Un Pledge tiene N Opportunities (una por período), y cada Opportunity tiene
1+ Payments (intentos: el primero puede fallar y reintentarse).

REGLAS DE NEGOCIO QUE APLICA ESTE SCRIPT
----------------------------------------
1. **Deduplicación de payments por payment_id**: si el export del CRM tiene
   filas duplicadas exactas del mismo Payment ID, se conserva solo una.
   Es una salvaguarda contra errores del export, NO descarta intentos
   legítimos.

2. **Filtro de "huérfanos"**: payments cuyo `crm_donation_id` no resuelve
   a un Pledge se DESCARTAN. Estos suelen ser donaciones únicas (one-off)
   que están fuera del scope del modelo (solo nos interesan donantes
   recurrentes). En la última corrida productiva esto filtró ~169k filas
   (~3% del total).

3. **Target del Motor 1 (is_success)**: definido como `status == "Paid"`.
   La columna fuente es `Status` del CSV de payments (col 34), que tiene
   valores `Paid` y `Payment Failed`. Cualquier otro valor (nulo, refunded,
   etc.) se clasifica como NO éxito.

4. **Anchor temporal `attempt_ts`**: se construye con jerarquía de fallback
   `payment_date → scheduled_date`. Si ambas son null, la fila NO se
   descarta automáticamente pero queda registrada en el CSV de errores
   (sección 'no_date_resolved'). Semántica:
       - payment_date  = fecha efectiva del cobro
       - scheduled_date = fecha programada del cobro
   Las dos tienen semánticas compatibles; cuando payment_date no está
   poblada, scheduled_date es el mejor proxy disponible.

5. **Ordenamiento cronológico estricto**: se ordena por
   `(pledge_id, attempt_ts, payment_id)` ANTES de calcular features de
   secuencia. Esto es CRÍTICO para evitar data leakage — las features
   acumuladas (streaks, success_count_cum) usan `.shift(1)` para incluir
   solo información ESTRICTAMENTE anterior al intento actual.

ENTRADA
-------
- data/unified/payments/payments_unified.parquet
- data/unified/opportunity/opportunity_unified.parquet
- data/unified/pledge/pledge_unified.parquet

SALIDA
------
- data/staging/stg_payment_attempts.parquet (~5.3M filas × 17 cols)
  Esta es la "tabla canónica" del proyecto. Todo lo que viene después
  la lee como insumo.

- data/staging/staging_quality_report.json
  Reporte de calidad estructurado (datos crudos para procesamiento programático).

- data/staging/staging_quality_report.txt
  Reporte de calidad en formato tabla legible (MÉTRICA | VALOR | INTERPRETACIÓN),
  con alertas automáticas si algún indicador supera el umbral aceptable.

- data/staging/staging_errors.csv
  Inventario de filas filtradas o problemáticas, con columna `error_type`:
      orphans          → payments sin pledge_id post-join
      duplicates       → payment_id duplicado eliminado
      ambiguous_status → status distinto de 'Paid' / 'Payment Failed'
      no_date_resolved → sin Payment Date ni Scheduled Date utilizable
  Permite auditoría manual de los registros que el pipeline no procesó.

NOTA DE TRAZABILIDAD
--------------------
Los nombres de columna que aparecen acá ya son post-snake_case (ver
00_unify_raw_data.py). El mapeo `nombre original CSV → nombre snake_case`.

Flujo:
    Unified Parquet → Enriquecimiento (joins) → Feature Engineering Base → Staging Parquet
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configuración de rutas (relativas a la raíz del entregable v8)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
UNIFIED_DIR = DATA_DIR / "unified"
STAGING_DIR = DATA_DIR / "staging"

# Umbrales para alertas automáticas en el reporte de calidad.
# Si una métrica supera el umbral, se marca con ⚠️ ¡ALERTA! en la columna interpretación.
ALERT_THRESHOLDS = {
    "rows_no_pledge_pct":     5.0,   # % de huérfanos aceptable
    "rows_ambiguous_pct":     5.0,   # % de status ambiguo aceptable
    "rows_no_date_pct":       1.0,   # % de filas sin fecha
    "drop_rate_pct":         10.0,   # % total descartado
    "dropped_duplicates_pct": 0.5,   # duplicados deberían ser ~0
}

def ensure_dirs():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

def load_unified_data():
    """
    Carga las 3 tablas unificadas relevantes para el staging, leyendo SOLO
    las columnas necesarias (column pruning del Parquet).

    De 36 columnas en payments, 72 en opportunity y 117 en pledge, este
    pipeline solo necesita las siguientes:

    PAYMENTS (11 cols):
        payment_id       → PK del intento (de "Payment ID" col 29 del CSV)
        crm_donation_id  → FK a opportunity (de "CRM Donation ID" col 14)
        payment_date     → fecha primaria del intento (de "Payment Date" col 17)
        scheduled_date   → fecha fallback si payment_date es null (de "Scheduled Date" col 19)
        status           → DEFINE EL TARGET is_success (de "Status" col 34,
                           valores: "Paid" / "Payment Failed")
        payment_amount   → monto del intento (de "Payment Amount" col 16)
        payment_method   → medio de pago (de "Payment Method" col 18)
        bank_name        → banco emisor con mojibake (de "Bank Name" col 22)
        payment_gateway  → pasarela (de "Payment Gateway" col 28)
        payment_submethod → sub-tipo (de "Payment Submethod" col 31)
        payment_status   → cargado para auditoría (NO se usa; el target usa `status`)

    OPPORTUNITY (4 cols):
        id                          → PK, llave del join con payments.crm_donation_id
        npe03_recurring_donation_c  → FK al pledge (de "npe03__Recurring_Donation__c")
        contact_id                  → FK al donante (de "ContactId" del CSV)
        stage_name                  
                                       

    PLEDGE (3 cols):
        id            → PK del pledge
        npsp_status_c → status del pledge (de "npsp__Status__c")
        created_date  → fecha de creación; se renombra a `pledge_created_date`
                        para evitar colisión con `created_date` de payments en el join

    REPLICABILIDAD: si en el futuro se quiere agregar features adicionales
    (ej: pledge.npsp_start_date_c, opportunity.amount), basta con agregarlas
    a las listas pay_cols/opp_cols/pledge_cols.
    """
    print("[1/5] Cargando datos unificados...")

    # Payments: La base transaccional de intentos.
    # Cargamos solo 11 de las 36 columnas para ahorrar memoria y tiempo.
    # NOTA: scheduled_date reemplazó a created_date como fallback de fecha
    # (más coherente semánticamente — fecha programada vs fecha de creación
    # del registro en SF).
    pay_cols = [
        "payment_id", "crm_donation_id", "payment_date", "scheduled_date", "status",
        "payment_amount", "payment_method", "bank_name", "payment_gateway",
        "payment_submethod", "payment_status"
    ]
    df_pay = pd.read_parquet(UNIFIED_DIR / "payments/payments_unified.parquet", columns=pay_cols)

    # Opportunity: tabla puente. Su único rol funcional acá es conectar
    # Payment con Pledge y aportar contact_id. Ninguna otra columna de
    # Opportunity entra como feature al modelo.
    opp_cols = ["id", "npe03_recurring_donation_c", "contact_id", "stage_name"]
    df_opp = pd.read_parquet(UNIFIED_DIR / "opportunity/opportunity_unified.parquet", columns=opp_cols)

    # Pledge: solo necesitamos el id, el status y la fecha de creación.
    # Para Motor 2 más adelante se cargan también npsp_start_date_c y
    # npsp_end_date_c, pero acá no.
    pledge_cols = ["id", "npsp_status_c", "created_date"]
    df_pledge = pd.read_parquet(UNIFIED_DIR / "pledge/pledge_unified.parquet", columns=pledge_cols)
    # Renombrar `created_date` para que no colisione con el de payments en el merge.
    df_pledge = df_pledge.rename(columns={"created_date": "pledge_created_date"})

    return df_pay, df_opp, df_pledge

def clean_and_merge(df_pay, df_opp, df_pledge, report, errors):
    """
    Ejecuta la secuencia: dedup → join → filtro de huérfanos.

    Además del DataFrame procesado, actualiza:
      - `report` (dict): métricas de calidad (raw_payments, dropped_duplicates,
        rows_no_pledge_after_join, etc.)
      - `errors` (dict): captura las filas filtradas para el CSV de auditoría
        en las claves 'duplicates' y 'orphans'.

    REGLAS DE DEDUPLICACIÓN
    -----------------------
    `drop_duplicates(subset=["payment_id"])` elimina filas con el MISMO
    payment_id. Esto NO descarta intentos legítimos múltiples del mismo
    pledge — cada intento tiene su propio payment_id. Solo atrapa errores
    de export del CRM donde un mismo registro se duplica por accidente.

    Si no había duplicados (caso normal), `errors["duplicates"]` queda vacío.

    REGLAS DE JOIN
    --------------
    Ambos joins son LEFT (`how="left"`) — preservan todas las filas de
    payments aunque no haya match en opportunity o pledge. Los matches
    fallidos quedan con NA en las columnas del lado derecho, y se filtran
    al final con el filtro de huérfanos.

    REGLA DE HUÉRFANOS
    ------------------
    Un "huérfano" es un payment cuyo `crm_donation_id` no resuelve a un
    Pledge (porque la Opportunity no existe o la Opportunity no apunta a
    ningún Pledge). Estos suelen ser:
        - Donaciones únicas (one-off) que no entran al modelo de recurrentes
        - Errores referenciales del CRM

    Como el modelo predice comportamiento de donantes RECURRENTES, descartar
    huérfanos es coherente con el scope. En la última corrida productiva,
    este filtro eliminó ~169.593 filas (~3% del total).
    """
    print("[2/5] Limpiando y unificando entidades...")

    # Métrica inicial: cuántas filas de payments llegaron a esta etapa.
    report["raw_payments"] = len(df_pay)

    # === 1. Deduplicación estricta de Payments por payment_id ===
    # Identificamos duplicados (NO el primero, todas las apariciones extras)
    # y los CAPTURAMOS antes de eliminarlos, para incluirlos en el CSV de errores.
    dup_mask = df_pay.duplicated(subset=["payment_id"], keep="first")
    errors["duplicates"] = df_pay[dup_mask].copy()
    n_dups = int(dup_mask.sum())

    df_pay = df_pay[~dup_mask].copy()
    report["dropped_duplicates"] = n_dups
    report["rows_after_dedup"] = len(df_pay)
    print(f"   - Deduplicados en Payments: {n_dups:,} registros eliminados.")

    # === 2. Join Payments → Opportunity ===
    # Llave: payments.crm_donation_id (FK) → opportunity.id (PK).
    # Tipo: LEFT (preservamos todos los payments aunque no haya Opportunity).
    df = df_pay.merge(df_opp, left_on="crm_donation_id", right_on="id", how="left")

    # === 3. Join Resultado → Pledge ===
    # Llave: opportunity.npe03_recurring_donation_c (FK) → pledge.id (PK).
    # `suffixes=("", "_pledge")` resuelve la colisión del campo `id`:
    #   - El `id` de opportunity (sin sufijo) viene del join anterior y
    #     ya queda obsoleto (lo dropeamos abajo).
    #   - El `id` de pledge se renombra a `id_pledge` (también lo dropeamos).
    df = df.merge(df_pledge, left_on="npe03_recurring_donation_c", right_on="id", how="left", suffixes=("", "_pledge"))

    # === Limpieza post-merge ===
    # Dropear las claves foráneas de los lados derechos (ya no se necesitan).
    df = df.drop(columns=["id", "id_pledge"])
    # Renombrar la FK del pledge a un nombre legible y estable.
    df = df.rename(columns={"npe03_recurring_donation_c": "pledge_id"})

    # === 4. Filtro de huérfanos ===
    # Si `pledge_id` quedó NA después del join, este payment no pertenece
    # a un pledge válido → fuera del scope del modelo.
    # Capturamos los huérfanos ANTES de filtrarlos para el CSV de errores.
    orphan_mask = df["pledge_id"].isna()
    errors["orphans"] = df[orphan_mask].copy()
    n_orphans = int(orphan_mask.sum())
    report["rows_no_pledge_after_join"] = n_orphans

    df = df[~orphan_mask].copy()
    print(f"   - Huérfanos descartados: {n_orphans:,} registros sin pledge_id válido.")

    return df

def process_timestamps_and_status(df, report, errors):
    """
    Estandariza fechas y construye el TARGET del Motor 1 (`is_success`).
    Además captura filas problemáticas (sin fecha resoluble + status ambiguo).

    ANCHOR TEMPORAL `attempt_ts`
    ----------------------------
    Fecha del intento de cobro. Jerarquía de fallback (2 niveles):
        1. payment_date    (col 17 del CSV: "Payment Date")
        2. scheduled_date  (col 19 del CSV: "Scheduled Date") si payment_date es null

    Si ambas son null, `attempt_ts` queda NA. Estas filas se REPORTAN en
    `errors['no_date_resolved']` para auditoría, pero NO se eliminan del
    DataFrame (el ordenamiento las dejará al final con NaT, y aguas abajo
    pueden romper merge_asof — el caller decide qué hacer).

    TARGET DEL MOTOR 1 (`is_success`)
    ---------------------------------
    is_success = 1 ↔ status == "Paid"
    is_failure = 1 ↔ status == "Payment Failed"

    Cualquier OTRO valor de status (nulo, refunded, pending, written off,
    etc.) → is_success=0 e is_failure=0 (ambiguo). Estas filas SE REPORTAN
    en `errors['ambiguous_status']` para que el cliente pueda revisarlas
    y decidir si tipificarlas en SF.

    ⚠️ El modelo aprende a predecir literalmente el string "Paid". Si
    Salesforce cambia el vocabulario (ej: "PAID", "Completed"), el target
    colapsa a 0 para todas las filas. Por eso es crítico el reporte de
    `status_candidates` del paso 00 y este conteo de `ambiguous_status`.

    ORDENAMIENTO CRONOLÓGICO
    ------------------------
    Se ordena por (pledge_id, attempt_ts, payment_id) ANTES de calcular
    features de secuencia. Esto garantiza que `.shift(1)`, `.cumsum()`
    y operaciones similares usen información ESTRICTAMENTE anterior al
    intento actual — condición sine qua non para evitar leakage en
    streaks y conteos acumulados.

    El tie-break por payment_id se aplica cuando dos intentos tienen el
    mismo timestamp exacto (poco frecuente pero posible).
    """
    print("[3/5] Estandarizando tiempos y estados...")

    # === Anchor temporal con fallback de 2 niveles ===
    # 1) payment_date → primario
    # 2) scheduled_date → fallback (fecha programada del cobro)
    df["attempt_ts"] = pd.to_datetime(df["payment_date"], errors="coerce")
    n_payment_date_ok = int(df["attempt_ts"].notna().sum())

    # Para las filas donde payment_date es null, intentamos scheduled_date.
    fallback_mask = df["attempt_ts"].isna()
    df.loc[fallback_mask, "attempt_ts"] = pd.to_datetime(
        df.loc[fallback_mask, "scheduled_date"], errors="coerce"
    )
    n_used_fallback = int((fallback_mask & df["attempt_ts"].notna()).sum())

    # Capturar filas sin fecha resoluble (ambas null).
    no_date_mask = df["attempt_ts"].isna()
    errors["no_date_resolved"] = df[no_date_mask].copy()
    n_no_date = int(no_date_mask.sum())
    report["rows_no_date_resolved"] = n_no_date
    report["rows_used_scheduled_date_fallback"] = n_used_fallback
    report["rows_with_payment_date"] = n_payment_date_ok
    print(f"   - Fechas: {n_payment_date_ok:,} con payment_date, "
          f"{n_used_fallback:,} usaron scheduled_date como fallback, "
          f"{n_no_date:,} sin fecha resoluble.")

    df["pledge_start_ts"] = pd.to_datetime(df["pledge_created_date"], errors="coerce")

    # === Target del Motor 1: éxito vs fallo ===
    # Profiling de los datos reales:
    #   status="Paid"           → ~85% de filas (éxito de cobro)
    #   status="Payment Failed" → ~12% de filas (fallo explícito)
    #   status=null / otros     → ~3% de filas (ambiguos)
    # is_success es binaria (Paid vs todo lo demás).
    # is_failure marca solo fallos EXPLÍCITOS (excluye ambiguos).
    df["is_success"] = (df["status"] == "Paid").astype(int)
    df["is_failure"] = (df["status"] == "Payment Failed").astype(int)

    # === Captura de filas con status ambiguo ===
    # NO se filtran del staging (entrarían al modelo con is_success=0), pero
    # SÍ se registran en errors para auditoría. Saber cuántas filas tienen
    # status no clasificable es señal de calidad del CRM.
    ambiguous_mask = ~df["status"].isin(["Paid", "Payment Failed"])
    errors["ambiguous_status"] = df[ambiguous_mask].copy()
    n_ambiguous = int(ambiguous_mask.sum())
    report["rows_ambiguous_status"] = n_ambiguous
    print(f"   - Status ambiguo: {n_ambiguous:,} filas con status distinto de Paid/Payment Failed.")

    # === Ordenamiento cronológico estricto ===
    # CRÍTICO para que las features de secuencia (streaks, cumulativos)
    # respeten el orden temporal y no introduzcan leakage.
    # Filas con attempt_ts=NaT quedan al final del orden.
    df = df.sort_values(["pledge_id", "attempt_ts", "payment_id"]).reset_index(drop=True)

    return df

def add_sequence_features(df):
    """
    Genera las features de SECUENCIA HISTÓRICA por pledge. Estas son
    features "tipo ventana" que para cada intento miran el pasado del
    mismo pledge.

    INVARIANTE FUNDAMENTAL (anti-leakage):
        Toda feature acá usa `.shift(1)` para que el intento N use
        información SOLO de los intentos 1..N-1, NUNCA de N en adelante.

    FEATURES GENERADAS:
        - attempt_n              → posición del intento en la secuencia del pledge (1, 2, 3, ...)
        - last_attempt_ts        → fecha del intento previo (auxiliar)
        - days_since_last_attempt → tiempo desde el intento previo (en días)
                                    Probablemente el predictor #1 del Motor 1.
        - success_count_cum      → éxitos ACUMULADOS antes del intento actual
        - failure_count_cum      → fallos ACUMULADOS antes del intento actual
        - pre_failure_streak     → racha consecutiva de fallos hasta el intento
                                   anterior (resetea al primer éxito)

    El primer intento de cada pledge queda con NA en days_since_last_attempt
    (porque no hay anterior). Las features acumuladas quedan en 0.
    """
    print("[4/5] Generando variables de secuencia (Motor 1)...")

    # Agrupamos por pledge — todas las features acumuladas se calculan
    # DENTRO del grupo, sin cruzar pledges.
    groups = df.groupby("pledge_id")

    # 1. Número de intento dentro del pledge (1-indexed por convención).
    df["attempt_n"] = groups.cumcount() + 1

    # 2. Tiempo desde el último intento.
    # shift(1) saca el attempt_ts del intento INMEDIATAMENTE anterior.
    df["last_attempt_ts"] = groups["attempt_ts"].shift(1)
    df["days_since_last_attempt"] = (df["attempt_ts"] - df["last_attempt_ts"]).dt.total_seconds() / 86400

    # 3. Conteos acumulados de éxitos y fallos previos.
    # El patrón es: shift(1) para excluir el intento actual, fillna(0)
    # para el primer intento de cada pledge, cumsum() para acumular.
    df["success_count_cum"] = groups["is_success"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    df["failure_count_cum"] = groups["is_failure"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    
    # 4. Streak de fallas: cuántos fallos CONSECUTIVOS lleva el pledge
    #    inmediatamente antes del intento actual. Resetea a 0 con cada éxito.
    #
    #    Ejemplo de secuencia para un pledge (cronológicamente):
    #        is_success:    [0, 0, 1, 0, 0, 0, 1, 0]
    #        pre_failure_streak (lo que queremos):
    #                       [0, 1, 2, 0, 1, 2, 3, 0]
    #        Lectura: al intento 7 (índice 6, valor 1) llegamos con 3 fallos
    #        consecutivos previos (intentos 4, 5, 6). El intento 8 ve la
    #        secuencia reseteada porque hubo un éxito justo antes.
    #
    #    Por qué importa: en la práctica, una racha de 3+ fallos consecutivos
    #    es uno de los predictores más fuertes de fallo del siguiente intento.

    # `streak_id` queda como variable auxiliar/diagnóstico — no se usa
    # downstream, pero ayuda a entender la lógica del streak.
    is_success = df["is_success"]
    df["streak_id"] = groups["is_success"].transform(lambda x: x.shift(1).fillna(0).cumsum())

    def fast_streak(x):
        """
        Calcula la racha de fallos consecutivos para una secuencia.

        Itera elemento a elemento (no es vectorizable trivialmente porque
        cada valor depende del anterior + el reset condicional). Para los
        ~5M intentos del staging tarda <30s.

        Devuelve un array DESPLAZADO 1 posición — es decir, el valor en
        la posición i es la racha al inicio del intento i (sin contar el
        intento i mismo). Esto preserva la regla anti-leakage.
        """
        s = []
        curr = 0
        for val in x:
            if val == 0:   # is_success=0 → fallo → la racha crece
                curr += 1
            else:          # is_success=1 → éxito → la racha se resetea
                curr = 0
            s.append(curr)
        # Shift para que el valor en posición i represente el estado ANTES
        # del intento i (no después). El primer intento de cada pledge
        # queda en 0 (no hay racha previa).
        res = pd.Series(s).shift(1).fillna(0).values
        return res

    df["pre_failure_streak"] = groups["is_success"].transform(fast_streak)

    return df

def finalize_quality_report(report, n_final):
    """
    Calcula las métricas derivadas (porcentajes y tasas) a partir de los
    conteos crudos acumulados durante el procesamiento.

    Debe llamarse DESPUÉS de que todas las etapas (clean_and_merge,
    process_timestamps_and_status, add_sequence_features) hayan corrido,
    y antes de imprimir/persistir el reporte.
    """
    raw = max(report.get("raw_payments", 0), 1)  # evitar div/0
    report["final_rows"] = int(n_final)

    # Tasa global de drop: cuánto se descartó vs lo que entró.
    report["drop_rate_pct"] = round(100 * (raw - n_final) / raw, 2)

    # Porcentajes específicos por categoría de error.
    report["rows_no_pledge_pct"] = round(100 * report.get("rows_no_pledge_after_join", 0) / raw, 2)
    report["rows_ambiguous_pct"] = round(100 * report.get("rows_ambiguous_status", 0) / raw, 2)
    report["rows_no_date_pct"] = round(100 * report.get("rows_no_date_resolved", 0) / raw, 2)
    report["dropped_duplicates_pct"] = round(100 * report.get("dropped_duplicates", 0) / raw, 2)

    return report


def print_and_save_quality_report(report, df_final, output_dir):
    """
    Imprime un reporte de calidad estructurado tipo tabla (formato del
    screenshot acordado con el cliente) y lo guarda en 2 formatos:

        - staging_quality_report.json → métricas crudas (legible por máquina)
        - staging_quality_report.txt  → tabla con interpretaciones (legible por humano)

    El reporte sigue el formato:

        MÉTRICA                       | VALOR     | INTERPRETACIÓN
        ------------------------------+-----------+-------------------------
        Raw payments                  | 5,566,032 | Total recibido del archivo original.
        Dropped duplicates            |         0 | Duplicados de payment_id eliminados.
        ...

    Métricas que superen umbrales de ALERT_THRESHOLDS se marcan con ⚠️ ¡ALERTA!.

    Calcula tasa_exito real (is_success.mean()) sobre el DataFrame final
    para reportarla en la tabla.
    """
    # success_rate sobre el dataset final (intentos válidos).
    if "is_success" in df_final.columns and len(df_final) > 0:
        report["success_rate_pct"] = round(100 * float(df_final["is_success"].mean()), 2)
    else:
        report["success_rate_pct"] = 0.0

    # Definición ordenada de las métricas del reporte:
    # (key_en_report, label_humano, formato, interpretación_base)
    METRICS = [
        ("raw_payments",                "Raw payments",                "int",
         "Total de registros recibidos del archivo original."),
        ("dropped_duplicates",          "Dropped duplicates",          "int",
         "Filas con payment_id duplicado eliminadas (errores de export)."),
        ("rows_after_dedup",            "Rows after dedup",            "int",
         "Registros únicos tras deduplicación por payment_id."),
        ("rows_no_pledge_after_join",   "Rows no pledge after join",   "int",
         "Huérfanos: payments sin match en Opportunity → Pledge."),
        ("rows_no_date_resolved",       "Rows no date resolved",       "int",
         "Filas sin Payment Date ni Scheduled Date utilizable."),
        ("rows_used_scheduled_date_fallback", "Rows w/ scheduled fallback", "int",
         "Filas que usaron Scheduled Date como fallback de Payment Date."),
        ("rows_ambiguous_status",       "Rows ambiguous status",       "int",
         "Status distinto de Paid/Payment Failed (no clasificable)."),
        ("final_rows",                  "Final rows",                  "int",
         "Registros finales que pasaron todos los filtros."),
        ("drop_rate_pct",               "Drop rate pct",               "pct",
         "Porcentaje total de filas descartadas en la limpieza."),
        ("success_rate_pct",            "Success rate pct",            "pct",
         "Efectividad de cobro (is_success / total) en el dataset final."),
        ("rows_no_pledge_pct",          "Rows no pledge pct",          "pct",
         "Proporción de huérfanos respecto al total original."),
        ("rows_ambiguous_pct",          "Rows ambiguous pct",          "pct",
         "Proporción de filas con status ambiguo respecto al total original."),
        ("rows_no_date_pct",            "Rows no date pct",            "pct",
         "Proporción de filas sin fecha respecto al total original."),
    ]

    lines = []
    sep = "=" * 110
    sub = "-" * 110
    lines.append(sep)
    lines.append("REPORTE DE CALIDAD — STAGING DE PAGOS")
    lines.append(sep)
    lines.append(f"{'MÉTRICA':<32} | {'VALOR':>12} | INTERPRETACIÓN")
    lines.append(sub)

    for key, label, fmt, interp in METRICS:
        v = report.get(key, 0)
        # Formateo del valor según tipo.
        if fmt == "pct":
            val_str = f"{float(v):.2f}%"
        else:
            val_str = f"{int(v):,}"

        # Chequeo de umbral → prefijo de alerta.
        threshold = ALERT_THRESHOLDS.get(key)
        if threshold is not None and float(v) > threshold:
            interp_final = f"¡ALERTA! {interp} (umbral: {threshold})"
        else:
            interp_final = interp

        lines.append(f"{label:<32} | {val_str:>12} | {interp_final}")

    lines.append(sep)
    txt_output = "\n".join(lines)

    # === Imprimir a consola ===
    print()
    print(txt_output)
    print()

    # === Guardar a archivos ===
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "staging_quality_report.txt"
    json_path = output_dir / "staging_quality_report.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_output + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"   Reporte guardado en:")
    print(f"     - {txt_path}")
    print(f"     - {json_path}")


def save_errors_csv(errors, output_dir):
    """
    Consolida todas las filas filtradas o problemáticas en un único CSV
    con una columna `error_type` que identifica el motivo del registro.

    Categorías capturadas:
        - duplicates       → payment_id duplicado eliminado
        - orphans          → sin pledge_id post-join (descartado del staging)
        - no_date_resolved → sin Payment Date ni Scheduled Date
        - ambiguous_status → status distinto de Paid/Payment Failed

    Si no hay errores en ninguna categoría, NO se genera el archivo
    (para no dejar CSVs vacíos confundiendo al cliente).

    Caso BORDE: 'orphans' contiene columnas adicionales del merge (status_,
    payment_status_, etc.) que no tienen las otras categorías. Pandas
    maneja esto rellenando con NaN al concatenar.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "staging_errors.csv"

    pieces = []
    for error_type, error_df in errors.items():
        if error_df is None:
            continue
        if not isinstance(error_df, pd.DataFrame) or len(error_df) == 0:
            continue
        # Etiquetamos cada bloque con su tipo de error.
        tagged = error_df.copy()
        tagged.insert(0, "error_type", error_type)
        pieces.append(tagged)

    if not pieces:
        print(f"   [INFO] Sin errores que reportar — no se genera staging_errors.csv")
        return

    combined = pd.concat(pieces, ignore_index=True, sort=False)
    combined.to_csv(csv_path, index=False)
    print(f"   CSV de errores guardado: {csv_path} ({len(combined):,} filas)")
    # Desglose por categoría para visibilidad rápida en consola.
    print(f"     Desglose por tipo:")
    for et, n in combined["error_type"].value_counts().items():
        print(f"       - {et}: {n:,}")


def save_staging(df):
    """
    Persiste el staging final con las 17 columnas que usan los pipelines de
    features aguas abajo.

    COLUMNAS FINALES (17):
        Identificadores:        payment_id, pledge_id, contact_id, crm_donation_id
        Anchor temporal:        attempt_ts
        Target & label aux:     is_success, is_failure
        Atributos del intento:  payment_amount, payment_method, bank_name,
                                payment_gateway, payment_submethod
        Features de secuencia:  attempt_n, days_since_last_attempt,
                                pre_failure_streak, success_count_cum,
                                failure_count_cum

    NOTA: las columnas `status`, `payment_status`, `payment_date`, `created_date`,
    `pledge_created_date`, `pledge_start_ts`, `last_attempt_ts`, `streak_id`
    NO se persisten — se usaron solo durante el procesamiento y ya
    cumplieron su función al derivar las features finales.

    """
    print("[5/5] Guardando dataset de Staging...")

    # Selección de columnas finales para el dataset transaccional canónico.
    final_cols = [
        "payment_id", "pledge_id", "contact_id", "crm_donation_id",
        "attempt_ts", "is_success", "is_failure", "payment_amount", "payment_method",
        "bank_name", "payment_gateway", "payment_submethod",
        "attempt_n", "days_since_last_attempt", "pre_failure_streak",
        "success_count_cum", "failure_count_cum"
    ]

    # Validación defensiva: solo conservamos las columnas que existen.
    existing_cols = [c for c in final_cols if c in df.columns]
    stg_df = df[existing_cols].copy()

    output_path = STAGING_DIR / "stg_payment_attempts.parquet"
    stg_df.to_parquet(output_path, index=False)

    print(f"\n[OK] Staging completado: {len(stg_df):,} intentos procesados.")
    print(f"Archivo generado: {output_path}")

def main():
    """
    Orquestador del staging. Mantiene dos acumuladores que se pasan a las
    funciones de transformación:

      - `report` (dict): contadores y porcentajes para el reporte de calidad.
      - `errors` (dict): DataFrames con las filas filtradas/problemáticas,
        que se consolidan al final en un único CSV de auditoría.

    Al finalizar, genera 4 artefactos en `data/staging/`:
        - stg_payment_attempts.parquet      → dataset canónico para el modelo
        - staging_quality_report.json       → métricas crudas
        - staging_quality_report.txt        → tabla legible con interpretaciones
        - staging_errors.csv                → filas descartadas / ambiguas
    """
    ensure_dirs()

    # Acumuladores compartidos entre las etapas.
    report = {}
    errors = {
        "duplicates":       pd.DataFrame(),
        "orphans":          pd.DataFrame(),
        "no_date_resolved": pd.DataFrame(),
        "ambiguous_status": pd.DataFrame(),
    }

    df_pay, df_opp, df_pledge = load_unified_data()

    df = clean_and_merge(df_pay, df_opp, df_pledge, report, errors)
    df = process_timestamps_and_status(df, report, errors)
    df = add_sequence_features(df)

    save_staging(df)

    # === Calcular métricas derivadas y emitir reporte + CSV de errores ===
    report = finalize_quality_report(report, n_final=len(df))
    print_and_save_quality_report(report, df, STAGING_DIR)
    save_errors_csv(errors, STAGING_DIR)


if __name__ == "__main__":
    main()
