#!/usr/bin/env python3
"""
02_enrich_with_contact.py — Features Motor 1 (paso 2 de 3)

PROPÓSITO
---------
Enriquecer el staging con atributos demográficos del donante:
        - state_norm        → departamento de residencia normalizado
        - gender_norm       → M/F/U
        - age_band          → banda etaria (<25, 25-35, ..., 65+)
        - ses_level         → nivel socioeconómico
        - contact_age_days  → antigüedad del registro del donante en el CRM

Estos atributos vienen de la tabla `contact` y se joinean al staging por
contact_id. 

REGLAS DE NEGOCIO IMPLEMENTADAS
-------------------------------
1. **state_norm**: normaliza el campo `MailingState` del CRM:
   - Strip de tildes y mayúsculas (NFKD).
   - "BOGOTÁ DC", "BOGOTÁ D.C.", "BOGOTA, D.C." → todos a "BOGOTA".
   - Long-tail trimmed: solo se conservan los TOP 25 departamentos
     más frecuentes + "UNKNOWN"; el resto se agrupa como "OTHER".
     Esto evita explosión de cardinalidad en la categórica.

2. **gender_norm**: M / F / U. Regla de fallback:
   - Primero intenta con `Gender__c` (campo estándar de SF).
   - Si null, intenta con `sescore__Gender__c` (campo de la app sescore).
   - Si ambos null → "U" (Unknown).
   - Se toma la PRIMERA LETRA en mayúscula. Si no es M ni F → "U".
   
3. **age_band**: bandas etarias a partir de `sescore__Age__c` 
   Bins: [-1, 25, 35, 45, 55, 65, 200] → labels ["<25", "25-35", "35-45",
                                                 "45-55", "55-65", "65+"]

4. **ses_level**: directo de `CCO_Socioeconomic_Level__c` (con fillna a
   "UNKNOWN"). Cobertura típica: ~20-30% de los donantes (el campo está
   pobremente poblado en el CRM).

5. **contact_age_days**: edad EN DÍAS del REGISTRO en el CRM
   (no edad biológica del donante).
   Definición: `attempt_ts - CreatedDate del contact`

OPTIMIZACIÓN DE MEMORIA
-----------------------
- Carga slim de contact (7 cols de 158).
- Conversión a categorías de pandas para columnas de baja cardinalidad
  (state_norm, gender_norm, age_band, ses_level, payment_gateway, etc.).
- Conversión de tipos numéricos a int8 / float32 / int32 según rango.
- `gc.collect()` después de borrar grandes DataFrames para liberar RAM.

Resultado: la tabla pasa de ~3GB en memoria a <1GB.

ENTRADA / SALIDA
----------------
Lee:    data/staging/stg_payment_attempts.parquet (cols slim)
        data/unified/contact/contact_unified.parquet (cols slim)
        data/features/bank_canonical.parquet
Genera: data/features/motor1_features.parquet (5.3M filas × 20 cols)

NOTA v8 — MÉTODO DE PAGO
------------------------
Se incorpora `payment_method` (medio de pago del intento) al feature set.
El cliente indicó que `payment_submethod` está frecuentemente vacío mientras
`payment_method` está poblado en ~99.999% de los intentos y con categorías
limpias (Auto Debit (ACH), Credit Card, Debit Card, ...). Ambas columnas se
persisten en motor1_features; el modelo (03_train/01_train_motor1.py) usa
`payment_method` en lugar de `payment_submethod`.
"""
from pathlib import Path
import re
import unicodedata
import gc
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAGING_V1 = ROOT / "data/staging/stg_payment_attempts.parquet"
CONTACT = ROOT / "data/unified/contact/contact_unified.parquet"
BANK_MAP = ROOT / "data/features/bank_canonical.parquet"
OUT = ROOT / "data/features/motor1_features.parquet"

TOP_N_STATES = 25  # Trim long-tail


def normalize_state(s):
    """
    Normaliza el departamento de residencia del donante.

    Reglas:
    1. Si nulo/vacío → "UNKNOWN".
    2. NFKD + ASCII para quitar tildes ("BOGOTÁ" → "BOGOTA").
    3. Upper case.
    4. Cualquier variante que contenga "BOGOT" se colapsa a "BOGOTA"
       (cubre "BOGOTÁ", "BOGOTÁ DC", "BOGOTA D.C.", "BOGOTA, D.C.", etc.).
    5. Limpieza de puntuación (comas, paréntesis).

    Devuelve string ya normalizado. La poda al top-25 ocurre en el caller.
    """
    if not isinstance(s, str) or not s:
        return "UNKNOWN"
    t = unicodedata.normalize("NFKD", s).encode("ascii", errors="ignore").decode("ascii").upper().strip()
    # Atajo: cualquier variante de Bogotá se unifica. Es de lejos el
    # departamento más frecuente y tiene más variantes de escritura.
    if "BOGOT" in t:
        return "BOGOTA"
    t = re.sub(r"[,\(\)]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    print("[1/5] Cargando Contact (slim)...")
    # Solo 7 columnas de las 158 disponibles. Mapeo a campos del CSV:
    #   id                          ← "Id" (col 103)
    #   mailing_state               ← "MailingState" (col 84)
    #   gender_c                    ← "Gender__c" (col 106)
    #   sescore_gender_c            ← "sescore__Gender__c" (col 48) — fallback
    #   cco_socioeconomic_level_c   ← "CCO_Socioeconomic_Level__c" (col 122)
    #   sescore_age_c               ← "sescore__Age__c" (col 72) — edad pre-calculada
    #   created_date                ← "CreatedDate" (col 113) — para contact_age_days
    contact_cols = ["id", "mailing_state", "gender_c", "sescore_gender_c",
                    "cco_socioeconomic_level_c", "sescore_age_c", "created_date"]
    contact = pd.read_parquet(CONTACT, columns=contact_cols)
    contact = contact.rename(columns={"id": "contact_id"})

    # === state_norm ===
    # Normaliza + trim al top-25 para evitar explosión de categorías.
    # Estados fuera del top quedan agrupados como "OTHER".
    contact["state_norm"] = contact["mailing_state"].apply(normalize_state)
    top_states = contact["state_norm"].value_counts().head(TOP_N_STATES).index.tolist()
    if "UNKNOWN" not in top_states:
        top_states.append("UNKNOWN")
    contact.loc[~contact["state_norm"].isin(top_states), "state_norm"] = "OTHER"
    contact["state_norm"] = contact["state_norm"].astype("category")
    print(f"  state_norm: {contact['state_norm'].nunique()} categorías")

    # === gender_norm ===
    # Regla de fallback: gender_c (estándar) → sescore_gender_c (custom) → "U".
    # Se toma solo la primera letra en mayúscula. Si no es M ni F → "U".
    contact["gender_norm"] = contact["gender_c"].fillna(contact["sescore_gender_c"]).fillna("U").astype(str).str.upper().str[0]
    contact.loc[~contact["gender_norm"].isin(["M", "F"]), "gender_norm"] = "U"
    contact["gender_norm"] = contact["gender_norm"].astype("category")

    # === age_band ===
    age_num = pd.to_numeric(contact["sescore_age_c"], errors="coerce")
    contact["age_band"] = pd.cut(age_num, bins=[-1, 25, 35, 45, 55, 65, 200],
                                  labels=["<25","25-35","35-45","45-55","55-65","65+"]).astype(str)
    contact.loc[contact["age_band"].isin(["nan", "NaN"]), "age_band"] = "UNKNOWN"
    contact["age_band"] = contact["age_band"].astype("category")

    # === ses_level ===
    # Nivel socioeconómico — sin transformación, solo fillna.
    contact["ses_level"] = contact["cco_socioeconomic_level_c"].fillna("UNKNOWN").astype(str).astype("category")

    # === Timestamp de creación del registro del donante (para contact_age_days) ===
    # Define la antigüedad del REGISTRO, no la edad biológica.
    contact["contact_created_ts"] = pd.to_datetime(contact["created_date"], errors="coerce", utc=True).dt.tz_localize(None)

    contact_slim = contact[["contact_id", "state_norm", "gender_norm", "age_band", "ses_level", "contact_created_ts"]].copy()
    del contact
    gc.collect()
    print(f"  Contact slim: {len(contact_slim):,}, mem={contact_slim.memory_usage(deep=True).sum()/1e6:.0f}MB")

    print("[2/5] Cargando staging v1 (cols slim)...")
    # v8: se agrega `payment_method` (medio de pago) además de payment_submethod.
    base_cols = ["payment_id", "pledge_id", "contact_id", "attempt_ts", "is_success",
                 "payment_amount", "payment_gateway", "payment_method", "payment_submethod",
                 "days_since_last_attempt", "pre_failure_streak",
                 "success_count_cum", "failure_count_cum", "attempt_n"]
    df = pd.read_parquet(STAGING_V1, columns=base_cols)
    # Optimizar dtypes
    df["is_success"] = df["is_success"].astype("int8")
    for c in ["pre_failure_streak", "success_count_cum", "failure_count_cum", "days_since_last_attempt"]:
        df[c] = df[c].astype("float32")
    df["attempt_n"] = df["attempt_n"].astype("int32")
    for c in ["payment_gateway", "payment_method", "payment_submethod"]:
        df[c] = df[c].astype("category")
    print(f"  staging filas: {len(df):,}, mem={df.memory_usage(deep=True).sum()/1e6:.0f}MB")

    print("[3/5] Aplicando bank_canonical...")
    # bank_canonical fue generado por 01_clean_bank_normalization.py.
    # Tiene EL MISMO ORDEN DE FILAS que el staging (porque ese script leyó
    # el staging entero y devolvió una columna paralela). Por eso se
    # asigna por `.values` directo, no por merge — es más rápido y no
    # consume el contact_id como key.
    bank_df = pd.read_parquet(BANK_MAP)
    df["bank_canonical"] = bank_df["bank_canonical"].values
    del bank_df
    gc.collect()

    print("[4/5] Merge con Contact...")
    # LEFT join: preservamos todos los intentos aunque el donante no
    # tenga registro en Contact. Esos casos quedan con NA en state/gender/etc.
    # y se rellenan con "UNKNOWN" abajo.
    df = df.merge(contact_slim, on="contact_id", how="left")
    del contact_slim
    gc.collect()
    # Fillna en categóricas (necesario porque merge inserta NaN)
    for c in ["state_norm", "gender_norm", "age_band", "ses_level"]:
        if df[c].dtype.name != "category":
            df[c] = df[c].astype("category")
        if "UNKNOWN" not in df[c].cat.categories and "U" not in df[c].cat.categories:
            df[c] = df[c].cat.add_categories(["UNKNOWN"])
        fill = "U" if c == "gender_norm" else "UNKNOWN"
        if fill not in df[c].cat.categories:
            df[c] = df[c].cat.add_categories([fill])
        df[c] = df[c].fillna(fill)

    print(f"  cobertura state ≠ UNKNOWN: {(df['state_norm'].astype(str) != 'UNKNOWN').mean()*100:.1f}%")
    print(f"  cobertura gender ≠ U: {(df['gender_norm'].astype(str) != 'U').mean()*100:.1f}%")
    print(f"  cobertura age ≠ UNKNOWN: {(df['age_band'].astype(str) != 'UNKNOWN').mean()*100:.1f}%")
    print(f"  cobertura ses ≠ UNKNOWN: {(df['ses_level'].astype(str) != 'UNKNOWN').mean()*100:.1f}%")

    print("[5/5] Calculando contact_age_days y guardando...")
    # contact_age_days = antigüedad del REGISTRO del donante en el CRM
    # al momento del intento. Construida como:
    #     attempt_ts (fecha del intento) - contact.CreatedDate (fecha de
    #     alta del registro)
    #
    # Si el resultado es negativo (intento anterior al alta del donante,
    # lo cual no debería pasar pero ocurre en datos sucios) → NA.
    df["contact_age_days"] = (df["attempt_ts"] - df["contact_created_ts"]).dt.days.astype("Int32")
    df.loc[df["contact_age_days"] < 0, "contact_age_days"] = pd.NA
    df = df.drop(columns=["contact_created_ts"])

    print(f"  cols finales: {df.columns.tolist()}")
    print(f"  filas: {len(df):,}, mem={df.memory_usage(deep=True).sum()/1e6:.0f}MB")
    df.to_parquet(OUT, index=False)
    print(f"[OK] {OUT}")


if __name__ == "__main__":
    main()
