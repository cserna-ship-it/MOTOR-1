#!/usr/bin/env python3
"""
07_build_allocation_features.py — Features Motor 1 (paso 3 de 3)

PROPÓSITO
---------
Construir features que capturen el HISTORIAL DE CAMBIOS CONTRACTUALES del
pledge para enriquecer Motor 1. La tabla `allocation` registra todos los
eventos del ciclo de vida de un pledge: downgrades de monto, cambios de
medio de pago, cambios de día de cobro, cambios de frecuencia, errores
operacionales, etc.

CONTEXTO DE NEGOCIO
-------------------
Un donante que ha tenido muchos cambios contractuales (especialmente
downgrades y errores) suele tener mayor probabilidad de fallar el próximo
cobro. Por ejemplo:
    - Un downgrade reciente indica deterioro de la capacidad de pago.
    - Un cambio de método de pago reciente introduce fricción operacional.
    - Errores acumulados son señal de problemas con la cuenta bancaria.

FEATURES GENERADAS (15 columnas)
--------------------------------
Conteos acumulados (totales antes del intento):
    alloc_n_total                     → todos los eventos
    alloc_n_downgrades_total          → solo downgrades
    alloc_n_method_changes_total      → cambios de medio de pago
    alloc_n_day_changes_total         → cambios de día de cobro
    alloc_n_frequency_changes_total   → cambios de frecuencia
    alloc_n_errors_total              → eventos en estado Error

Recencia:
    alloc_days_since_last → días desde el último evento (99999 si nunca hubo)

Flags de ventana 90 días (¿hubo un evento de tipo X en los últimos 90d?):
    alloc_recent90_downgrade
    alloc_recent90_method_change
    alloc_recent90_day_change
    alloc_recent90_frequency_change
    alloc_recent90_error

Estado contractual:
    alloc_last_type   → tipo del último evento (categórica)
    alloc_last_status → status del último evento (OK / Error)

REGLAS DE NEGOCIO IMPLEMENTADAS
-------------------------------
1. **Filtro `alloc_type != "Cancel Pledge"`**: los eventos de cancelación
   se descartan porque son post-churn y no aportan señal predictiva
   (además son redundantes con `npsp__EndDate__c`).

2. **Detección de cambios** se hace por inspección de los campos
   booleanos del CRM (`sescore__Frequency_Changed__c`, etc.) Y por
   detección de strings ("Downgrade" en alloc_type). Es belt-and-
   suspenders porque distintos eventos en SF se reflejan de forma
   distinta.

3. **Anti-leakage vía merge_asof con `direction="backward"` y
   `allow_exact_matches=False`**: para cada intento, se busca el último
   evento de allocation con `alloc_ts < attempt_ts` (estrictamente
   anterior). Si dos timestamps coinciden, el evento NO se considera —
   evita que el modelo "vea" eventos del mismo instante.

APROXIMACIÓN DE LAS FEATURES `recent90`
----------------------------------------
Las banderas de "¿hubo evento X en los últimos 90 días?" se aproximan con
`(alloc_days_since_last < 90 AND último_evento = X)`. La aproximación
falla cuando el último evento NO es del tipo X pero SÍ hubo uno antes
dentro de la ventana (ej: hace 30d un downgrade, hace 10d un method
change → la bandera de downgrade saldría 0 con esta aproximación).

La versión exacta requeriría 3 merge_asof adicionales, uno por tipo.
Se eligió la aproximación porque la pérdida de información es marginal
(<5% en pruebas) y la simplicidad/velocidad valen la pena.

ENTRADA / SALIDA
----------------
Lee:    data/unified/allocation/allocation_unified.parquet (8 cols)
        data/features/motor1_features.parquet (3 cols)
Genera: data/features/allocation_features.parquet (5.3M filas × 15 cols)

NOTA SOBRE STREAMING
--------------------
La escritura se hace en chunks de 1M filas porque el resultado completo
(5.3M × 15) podría no caber en RAM. Se usa pyarrow.ParquetWriter para
streaming.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ALLOCATION = ROOT / "data/unified/allocation/allocation_unified.parquet"
MOTOR1_FEATURES = ROOT / "data/features/motor1_features.parquet"
OUT = ROOT / "data/features/allocation_features.parquet"

T0 = time.time()
def t(msg): print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def main():
    t("Cargando Allocation...")
    # Cargamos solo 8 cols de las 67 disponibles en allocation_unified.
    # Mapeo a campos del CSV:
    #   sescore_recurring_donation_c    ← "sescore__Recurring_Donation__c"   (FK a pledge)
    #   sescore_processed_date_time_c   ← "sescore__Processed_Date_Time__c"  (timestamp del evento)
    #   sescore_type_c                  ← "sescore__Type__c"                 (tipo de evento)
    #   sescore_status_c                ← "sescore__Status__c"               (status del evento)
    #   sescore_old_payment_method_c    ← "sescore__Old_PaymentMethod__c"    (método anterior)
    #   sescore_new_payment_method_c    ← "sescore__New_PaymentMethod__c"    (método nuevo)
    #   sescore_day_of_month_changed_c  ← "sescore__Day_of_Month_Changed__c" (bool)
    #   sescore_frequency_changed_c     ← "sescore__Frequency_Changed__c"    (bool)
    cols = ["sescore_recurring_donation_c", "sescore_processed_date_time_c",
            "sescore_type_c", "sescore_status_c",
            "sescore_old_payment_method_c", "sescore_new_payment_method_c",
            "sescore_day_of_month_changed_c", "sescore_frequency_changed_c"]
    a = pd.read_parquet(ALLOCATION, columns=cols)

    # Renombre a nombres internos estables.
    a = a.rename(columns={"sescore_recurring_donation_c": "pledge_id",
                          "sescore_processed_date_time_c": "alloc_ts",
                          "sescore_type_c": "alloc_type",
                          "sescore_status_c": "alloc_status"})
    a["alloc_ts"] = pd.to_datetime(a["alloc_ts"], errors="coerce", utc=True).dt.tz_localize(None)

    # Filtrar registros sin timestamp o sin pledge_id (no se pueden ordenar
    # ni asociar al donante).
    a = a.dropna(subset=["alloc_ts", "pledge_id"])

    # Filtro de eventos de cancelación: estos son post-churn y no aportan
    # señal predictiva (la cancelación es lo que queremos predecir, no usar
    # como feature). Mantenerlos sería leakage.
    a = a[a["alloc_type"] != "Cancel Pledge"].copy()
    t(f"Allocations útiles: {len(a):,}")

    # === Flags binarios por tipo de evento ===
    # Estos se construyen con reglas distintas según cómo el CRM
    # representa cada tipo de cambio:

    # Downgrade: match en alloc_type por substring "Downgrade"
    # (cubre "Downgrade", "Downgrade Amount", etc.)
    a["is_downgrade"] = a["alloc_type"].astype(str).str.contains("Downgrade").astype("int32")

    # Cambio de método de pago: regla compuesta (OR de dos condiciones):
    #   a) alloc_type == "Payment Method Change", o
    #   b) los campos old/new PaymentMethod están ambos poblados y son distintos
    # La regla (b) atrapa cambios que no fueron tipificados explícitamente
    # como "Payment Method Change".
    a["is_method_change"] = ((a["alloc_type"] == "Payment Method Change") |
                              ((a["sescore_old_payment_method_c"].notna()) &
                               (a["sescore_new_payment_method_c"].notna()) &
                               (a["sescore_old_payment_method_c"] != a["sescore_new_payment_method_c"]))).astype("int32")

    # Cambios de día de cobro y frecuencia: vienen como booleanos en
    # el CRM, exportados como strings "true"/"false" (lowercase).
    a["is_day_change"] = (a["sescore_day_of_month_changed_c"].astype(str) == "true").astype("int32")
    a["is_frequency_change"] = (a["sescore_frequency_changed_c"].astype(str) == "true").astype("int32")

    # Eventos en estado de error. Se cuentan tanto "Error" como
    # "Error - Resolved" porque incluso los errores ya resueltos
    # son señal de fricción operacional pasada.
    a["is_error"] = a["alloc_status"].astype(str).isin(["Error", "Error - Resolved"]).astype("int32")
    a = a.drop(columns=["sescore_old_payment_method_c", "sescore_new_payment_method_c",
                         "sescore_day_of_month_changed_c", "sescore_frequency_changed_c"])

    t("Cumcounts por pledge...")
    a = a.sort_values(["pledge_id", "alloc_ts"]).reset_index(drop=True)
    g = a.groupby("pledge_id", sort=False)
    a["cum_n_alloc"] = g.cumcount() + 1
    for col in ["is_downgrade", "is_method_change", "is_day_change",
                "is_frequency_change", "is_error"]:
        a[f"cum_{col}"] = g[col].cumsum()

    t("Cargando staging slim...")
    stg = pd.read_parquet(MOTOR1_FEATURES, columns=["payment_id", "pledge_id", "attempt_ts"])
    stg["attempt_ts"] = stg["attempt_ts"].astype("datetime64[ns]")
    a["alloc_ts"] = a["alloc_ts"].astype("datetime64[ns]")
    t(f"Staging filas: {len(stg):,}")

    # Sort para merge_asof
    a = a.sort_values("alloc_ts").reset_index(drop=True)
    stg = stg.sort_values("attempt_ts").reset_index(drop=True)

    cum_cols = [c for c in a.columns if c.startswith("cum_")]
    flag_cols = ["is_downgrade", "is_method_change", "is_day_change", "is_frequency_change", "is_error"]
    asof_cols = ["pledge_id", "alloc_ts", "alloc_type", "alloc_status"] + cum_cols + flag_cols

    t("Merge_asof por chunks + escritura streaming...")
    import pyarrow as pa
    import pyarrow.parquet as pqw

    # Streaming write para no cargar 5.3M filas × 15 cols en RAM.
    # Cada chunk de 1M filas se procesa, escribe, y libera antes del siguiente.
    chunk_size = 1_000_000
    writer = None
    for start in range(0, len(stg), chunk_size):
        chunk = stg.iloc[start:start+chunk_size].copy()
        chunk = chunk.sort_values("attempt_ts").reset_index(drop=True)

        # merge_asof = JOIN temporal por proximidad. Para cada fila del
        # chunk (un intento), encuentra la fila de `a` con el mayor
        # `alloc_ts` que sea estrictamente menor que `attempt_ts`,
        # restringiendo a filas del mismo pledge_id.
        #
        # Parámetros críticos:
        #   - direction="backward": busca el evento ANTERIOR al intento
        #   - allow_exact_matches=False: si alloc_ts == attempt_ts, NO se
        #     considera. Anti-leakage para casos de timestamps idénticos.
        merged = pd.merge_asof(
            chunk, a[asof_cols],
            left_on="attempt_ts", right_on="alloc_ts",
            by="pledge_id", direction="backward", allow_exact_matches=False,
        )
        for c in cum_cols:
            merged[c] = merged[c].fillna(0).astype("int32")
        for c in flag_cols:
            merged[c] = merged[c].fillna(0).astype("int8")
        merged["alloc_days_since_last"] = ((merged["attempt_ts"] - merged["alloc_ts"]).dt.days).fillna(99999).astype("float32")
        is_recent_90 = merged["alloc_days_since_last"] < 90

        feat = pd.DataFrame({
            "payment_id": merged["payment_id"].values,
            "alloc_n_total": merged["cum_n_alloc"].values.astype("int32"),
            "alloc_n_downgrades_total": merged["cum_is_downgrade"].values.astype("int32"),
            "alloc_n_method_changes_total": merged["cum_is_method_change"].values.astype("int32"),
            "alloc_n_day_changes_total": merged["cum_is_day_change"].values.astype("int32"),
            "alloc_n_frequency_changes_total": merged["cum_is_frequency_change"].values.astype("int32"),
            "alloc_n_errors_total": merged["cum_is_error"].values.astype("int32"),
            "alloc_days_since_last": merged["alloc_days_since_last"].values,
            "alloc_recent90_downgrade": (is_recent_90 & (merged["is_downgrade"] == 1)).astype("int8").values,
            "alloc_recent90_method_change": (is_recent_90 & (merged["is_method_change"] == 1)).astype("int8").values,
            "alloc_recent90_day_change": (is_recent_90 & (merged["is_day_change"] == 1)).astype("int8").values,
            "alloc_recent90_frequency_change": (is_recent_90 & (merged["is_frequency_change"] == 1)).astype("int8").values,
            "alloc_recent90_error": (is_recent_90 & (merged["is_error"] == 1)).astype("int8").values,
            "alloc_last_type": merged["alloc_type"].fillna("NONE").astype(str).values,
            "alloc_last_status": merged["alloc_status"].fillna("NONE").astype(str).values,
        })
        table = pa.Table.from_pandas(feat)
        if writer is None:
            writer = pqw.ParquetWriter(OUT, table.schema)
        writer.write_table(table)
        t(f"  chunk {start:,}+{len(chunk):,} written")
    if writer:
        writer.close()
    t(f"OK → {OUT}")


if __name__ == "__main__":
    main()
