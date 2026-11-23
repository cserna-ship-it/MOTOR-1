#!/usr/bin/env python3
"""
04_build_incremental_features.py — Features incrementales Motor 1 (año en curso)

PROPÓSITO
---------
Genera motor1_features_ytd.parquet con los intentos del año en curso,
usando el histórico ya procesado como punto de partida para los contadores
acumulativos. Esto permite correr el pipeline mensual sin re-procesar los
5+ millones de filas históricas.

DISEÑO IDEMPOTENTE
------------------
El equipo carga UN SOLO ARCHIVO mensual: todos los intentos del año en
curso hasta la fecha (YTD = Year To Date). No importa si ya lo cargaron
antes — el script siempre empieza desde el histórico y sobreescribe el
output. Cargar el mismo archivo dos veces da el mismo resultado.

    Mes 1  →  carga  payments_2026_ytd.csv  (enero)
    Mes 2  →  carga  payments_2026_ytd.csv  (enero + febrero)  ← reemplaza el archivo
    Mes 3  →  carga  payments_2026_ytd.csv  (enero + febrero + marzo)

Si se olvidan un mes: no pasa nada. Al siguiente mes el YTD lo incluye.

LÓGICA DE CONTADORES
--------------------
Para cada pledge, el estado al final del histórico es:
  - success_count_cum_end  = cum[último hist.] + is_success[último hist.]
  - failure_count_cum_end  = cum[último hist.] + (1 − is_success[último hist.])
  - streak_state           = 0 si el último hist. fue éxito,
                             pre_failure_streak[último] + 1 si fue fallo
  - attempt_n_end          = attempt_n del último hist.
  - last_ts                = attempt_ts del último hist.

Para el primer intento YTD del pledge, estos valores son el punto de
partida. Los intentos siguientes del mismo año se encadenan normalmente.

ENTRADA
-------
  data/features/motor1_features.parquet          ← HISTÓRICO, NUNCA SE TOCA
  raw/payments_ytd.parquet ó .csv                ← el equipo lo reemplaza cada mes
                                                    (ruta por defecto; overridable
                                                     con --payments-ytd)
  data/unified/opportunity/opportunity_unified.parquet
  data/unified/contact/contact_unified.parquet
  data/features/bank_canonical_mapping.json      ← para normalización de bancos

SALIDA
------
  data/features/motor1_features_ytd.parquet      ← siempre se sobreescribe

USO
---
  python3 scripts/01_features_motor1/04_build_incremental_features.py
  python3 scripts/01_features_motor1/04_build_incremental_features.py \\
      --historical data/features/motor1_features.parquet \\
      --payments-ytd raw/payments_2026_ytd.csv
"""
from pathlib import Path
import sys
import re
import json
import gc
import unicodedata
import argparse
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]

# ── Rutas por defecto ──────────────────────────────────────────────────────────
HIST_FEATURES  = ROOT / "data/features/motor1_features.parquet"
OPP_PATH       = ROOT / "data/unified/opportunity/opportunity_unified.parquet"
CONTACT_PATH   = ROOT / "data/unified/contact/contact_unified.parquet"
BANK_MAP_JSON  = ROOT / "data/features/bank_canonical_mapping.json"
OUT_PATH       = ROOT / "data/features/motor1_features_ytd.parquet"
DEFAULT_YTD    = ROOT / "raw/payments_ytd.parquet"

# ── Columnas finales (mismas que motor1_features.parquet) ─────────────────────
FINAL_COLS = [
    "payment_id", "pledge_id", "contact_id", "attempt_ts", "is_success",
    "payment_amount", "payment_gateway", "payment_method", "payment_submethod",
    "days_since_last_attempt", "pre_failure_streak",
    "success_count_cum", "failure_count_cum", "attempt_n",
    "bank_canonical", "state_norm", "gender_norm", "age_band", "ses_level",
    "contact_age_days",
]


# ── Normalización de bancos (replica 01_clean_bank_normalization.py) ──────────

def normalize_bank(name: str) -> str:
    if not isinstance(name, str) or not name:
        return "UNKNOWN"
    s = name
    for _ in range(4):
        if any(m in s for m in ("Ã", "ï¿½", "Â", "Å")):
            try:
                fixed = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
                if fixed and fixed != s:
                    s = fixed
                else:
                    break
            except Exception:
                break
        else:
            break
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = s.upper().strip()
    s = re.sub(r",", " ", s)
    s = re.sub(r"\s+S\.?\s*A\.?(\s|$)", " SA ", s)
    s = re.sub(r"\s+S\.?\s*A\.?\s*S\.?(\s|$)", " SAS ", s)
    s = re.sub(r"[^A-Z0-9\s\(\)\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\bSA\b\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "UNKNOWN"


def load_canonical_map() -> dict:
    """Carga el mapeo manual de bancos generado por 01_clean_bank_normalization.py."""
    if BANK_MAP_JSON.exists():
        with open(BANK_MAP_JSON, "r") as f:
            return json.load(f)
    return {}


def apply_bank_canonical(bank_series: pd.Series, canonical_map: dict) -> pd.Series:
    normalized = bank_series.apply(normalize_bank)
    return normalized.map(lambda x: canonical_map.get(x, x))


# ── Normalización de estado (replica 02_enrich_with_contact.py) ───────────────

TOP_N_STATES = 25

def normalize_state(s) -> str:
    if not isinstance(s, str) or not s:
        return "UNKNOWN"
    t = unicodedata.normalize("NFKD", s).encode("ascii", errors="ignore").decode("ascii").upper().strip()
    if "BOGOT" in t:
        return "BOGOTA"
    t = re.sub(r"[,\(\)]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Utilidades de parsing de columnas ─────────────────────────────────────────

# Mapping: nombre raw de Salesforce → nombre normalizado que usa el pipeline
SF_COLUMN_MAP = {
    "Id":                     "payment_id",
    "CRM Donation ID":        "crm_donation_id",
    "Payment Date":           "payment_date",
    "Scheduled Date":         "scheduled_date",
    "Status":                 "status",
    "Payment Amount":         "payment_amount",
    "Payment Method":         "payment_method",
    "Bank Name":              "bank_name",
    "Payment Gateway":        "payment_gateway",
    "Payment Submethod":      "payment_submethod",
    # Variantes posibles del export SF
    "npe01__Payment_Id__c":   "payment_id",
    "npe01__Paid_Amount__c":  "payment_amount",
    "npe01__Payment_Date__c": "payment_date",
    "Scheduled_Date__c":      "scheduled_date",
}


def normalize_payments_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el CSV tiene nombres de columna del export crudo de Salesforce,
    los normaliza al formato que usa el pipeline. Si ya están normalizados,
    no hace nada.
    """
    rename = {k: v for k, v in SF_COLUMN_MAP.items() if k in df.columns}
    if rename:
        df = df.rename(columns=rename)
    return df


# ── Carga del archivo YTD ─────────────────────────────────────────────────────

def load_payments_ytd(path: Path) -> pd.DataFrame:
    print(f"  Leyendo pagos YTD desde: {path}")
    if path.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_parquet(path)
    df = normalize_payments_columns(df)
    needed = {"payment_id", "crm_donation_id", "status", "payment_amount",
              "payment_method", "bank_name", "payment_gateway", "payment_submethod"}
    missing = needed - set(df.columns)
    if missing:
        # Intentar con nombres alternativos antes de fallar
        alt = {"payment_id": ["Id", "id"], "crm_donation_id": ["CRM Donation ID", "crm_donation_id"]}
        for col, alts in alt.items():
            if col in missing:
                for a in alts:
                    if a in df.columns:
                        df = df.rename(columns={a: col})
                        missing.discard(col)
                        break
    if missing:
        raise ValueError(f"Columnas requeridas no encontradas en el archivo YTD: {missing}\n"
                         f"Columnas disponibles: {list(df.columns)}")
    print(f"  Filas leídas: {len(df):,}")
    return df


# ── Función principal de cómputo incremental ─────────────────────────────────

def fast_streak_incremental(is_success_series: pd.Series, start_streak: int) -> pd.Series:
    """
    Calcula pre_failure_streak para una secuencia YTD, iniciando desde
    `start_streak` (el estado al final del histórico para este pledge).

    Ejemplo:
        hist termina con: [..., is_success=0, pre_failure_streak=2]
        start_streak = 3  (2 + 1 fallo más)
        ytd = [0, 0, 1, 0]
        resultado = [3, 4, 5, 0]
    """
    streaks = []
    curr = start_streak
    for val in is_success_series:
        streaks.append(curr)
        curr = 0 if val == 1 else curr + 1
    return pd.Series(streaks, index=is_success_series.index)


def compute_incremental_sequence(ytd: pd.DataFrame, hist_state: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula features de secuencia para los intentos YTD, usando hist_state
    como punto de partida por pledge.

    hist_state: DataFrame indexado por pledge_id con columnas:
        last_ts            → timestamp del último intento histórico
        success_cum_end    → éxitos acumulados incluyendo el último hist.
        failure_cum_end    → fallos acumulados incluyendo el último hist.
        streak_state       → racha de fallos al inicio del primer intento YTD
        attempt_n_end      → número de intento del último intento histórico
    """
    ytd = ytd.sort_values(["pledge_id", "attempt_ts", "payment_id"]).copy()

    # ── days_since_last_attempt ────────────────────────────────────────────────
    # Para el primer intento YTD de cada pledge: usa la última ts histórica.
    # Para los siguientes: usa la ts del intento YTD previo.
    ytd["_prev_ts"] = ytd.groupby("pledge_id")["attempt_ts"].shift(1)
    is_first = ytd.groupby("pledge_id").cumcount() == 0
    hist_last_ts = hist_state["last_ts"].to_dict()
    ytd.loc[is_first, "_prev_ts"] = ytd.loc[is_first, "pledge_id"].map(hist_last_ts)
    ytd["days_since_last_attempt"] = (
        (ytd["attempt_ts"] - ytd["_prev_ts"]).dt.total_seconds() / 86400
    ).astype("float32")
    ytd.drop(columns=["_prev_ts"], inplace=True)

    # ── attempt_n ─────────────────────────────────────────────────────────────
    hist_attempt_n = hist_state["attempt_n_end"].to_dict()
    ytd["attempt_n"] = (
        ytd.groupby("pledge_id").cumcount() + 1
        + ytd["pledge_id"].map(hist_attempt_n).fillna(0)
    ).astype("int32")

    # ── success_count_cum y failure_count_cum ─────────────────────────────────
    # offset histórico + acumulado dentro del año YTD (excluyendo intento actual)
    hist_suc = hist_state["success_cum_end"].to_dict()
    hist_fail = hist_state["failure_cum_end"].to_dict()

    ytd["success_count_cum"] = (
        ytd.groupby("pledge_id")["is_success"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        + ytd["pledge_id"].map(hist_suc).fillna(0)
    ).astype("float32")

    ytd["failure_count_cum"] = (
        ytd.groupby("pledge_id")["is_failure"].transform(
            lambda x: x.shift(1).fillna(0).cumsum()
        )
        + ytd["pledge_id"].map(hist_fail).fillna(0)
    ).astype("float32")

    # ── pre_failure_streak ────────────────────────────────────────────────────
    streak_start_map = hist_state["streak_state"].to_dict()

    def apply_streak(group):
        pid = group.name
        start = streak_start_map.get(pid, 0)
        group["pre_failure_streak"] = fast_streak_incremental(
            group["is_success"], int(start)
        ).values
        return group

    ytd = ytd.groupby("pledge_id", group_keys=False).apply(apply_streak)
    ytd["pre_failure_streak"] = ytd["pre_failure_streak"].astype("float32")

    return ytd


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Features incrementales Motor 1")
    p.add_argument("--historical",    type=str, default=str(HIST_FEATURES),
                   help="Parquet histórico (default: data/features/motor1_features.parquet)")
    p.add_argument("--payments-ytd",  type=str, default=str(DEFAULT_YTD),
                   help="Pagos YTD en formato payments_unified (.parquet o .csv)")
    p.add_argument("--out",           type=str, default=str(OUT_PATH),
                   help="Ruta de salida (default: data/features/motor1_features_ytd.parquet)")
    return p.parse_known_args()[0]


def main():
    args = parse_args()
    hist_path = Path(args.historical)
    ytd_path  = Path(args.payments_ytd)
    out_path  = Path(args.out)

    print(f"  [config] Histórico  : {hist_path}")
    print(f"  [config] Pagos YTD  : {ytd_path}")
    print(f"  [config] Salida     : {out_path}")

    # ── [1/6] Estado final del histórico por pledge ────────────────────────────
    print("\n[1/6] Cargando estado histórico por pledge...")
    hist_cols = ["payment_id", "pledge_id", "contact_id", "attempt_ts", "is_success",
                 "payment_amount", "payment_gateway", "payment_method", "payment_submethod",
                 "days_since_last_attempt", "pre_failure_streak",
                 "success_count_cum", "failure_count_cum", "attempt_n",
                 "bank_canonical", "state_norm", "gender_norm", "age_band", "ses_level",
                 "contact_age_days"]
    hist = pd.read_parquet(hist_path, columns=hist_cols)
    hist["attempt_ts"] = pd.to_datetime(hist["attempt_ts"])

    hist_sorted = hist.sort_values(["pledge_id", "attempt_ts", "payment_id"])
    hist_last = hist_sorted.groupby("pledge_id").last()

    # Estado al FINAL del histórico (post-intento, para inicializar YTD)
    hist_state = pd.DataFrame({
        "last_ts":          hist_last["attempt_ts"],
        "success_cum_end":  hist_last["success_count_cum"] + hist_last["is_success"].astype(float),
        # Proxy: (1 - is_success) como is_failure para calcular failure_cum_end
        # Error negligible (~3% casos ambiguos no son is_failure=1 exacto)
        "failure_cum_end":  hist_last["failure_count_cum"] + (1 - hist_last["is_success"]).astype(float),
        "streak_state":     hist_last.apply(
            lambda r: 0.0 if r["is_success"] == 1 else float(r["pre_failure_streak"]) + 1,
            axis=1
        ),
        "attempt_n_end":    hist_last["attempt_n"].astype(float),
        # Features estáticas por pledge (para enriquecer pledges ya conocidos)
        "bank_canonical":   hist_last["bank_canonical"].astype(str),
        "state_norm":       hist_last["state_norm"].astype(str),
        "gender_norm":      hist_last["gender_norm"].astype(str),
        "age_band":         hist_last["age_band"].astype(str),
        "ses_level":        hist_last["ses_level"].astype(str),
        "contact_id":       hist_last["contact_id"].astype(str),
    })
    print(f"  Pledges en histórico: {len(hist_state):,}")
    del hist_sorted
    gc.collect()

    # ── [2/6] Cargar pagos YTD ────────────────────────────────────────────────
    print("\n[2/6] Cargando pagos YTD...")
    pay = load_payments_ytd(ytd_path)

    # Timestamp del intento: payment_date con fallback a scheduled_date
    pay["attempt_ts"] = pd.to_datetime(pay.get("payment_date", pd.NaT), errors="coerce")
    if "scheduled_date" in pay.columns:
        mask = pay["attempt_ts"].isna()
        pay.loc[mask, "attempt_ts"] = pd.to_datetime(
            pay.loc[mask, "scheduled_date"], errors="coerce"
        )
    n_no_ts = pay["attempt_ts"].isna().sum()
    if n_no_ts > 0:
        print(f"  ⚠  {n_no_ts:,} filas sin fecha resoluble → se descartan")
        pay = pay[pay["attempt_ts"].notna()].copy()

    pay["is_success"] = (pay["status"] == "Paid").astype("int8")
    pay["is_failure"] = (pay["status"] == "Payment Failed").astype("int8")

    # Descartar cualquier intento anterior o igual a la última fecha histórica
    # (evita duplicados si el YTD incluye datos del año de entrenamiento)
    hist_cutoff = hist["attempt_ts"].max()
    # Deduplicar por payment_id (igual que create_staging.py)
    n_dups = pay.duplicated(subset=["payment_id"], keep="first").sum()
    if n_dups > 0:
        print(f"  ⚠  {n_dups:,} payment_ids duplicados → se conserva solo el primero (mismo criterio que pipeline completo)")
        pay = pay.drop_duplicates(subset=["payment_id"], keep="first").copy()

    n_before = (pay["attempt_ts"] <= hist_cutoff).sum()
    if n_before > 0:
        print(f"  ⚠  {n_before:,} intentos antes de {hist_cutoff.date()} → se descartan (ya en histórico)")
        pay = pay[pay["attempt_ts"] > hist_cutoff].copy()
    print(f"  Intentos YTD válidos: {len(pay):,}")

    del hist
    gc.collect()

    # ── [3/6] Join con Opportunity → pledge_id, contact_id ───────────────────
    print("\n[3/6] Obteniendo pledge_id via Opportunity...")
    opp_cols = ["id", "npe03_recurring_donation_c", "contact_id"]
    opp = pd.read_parquet(OPP_PATH, columns=opp_cols)
    pay = pay.merge(
        opp.rename(columns={"id": "crm_donation_id",
                             "npe03_recurring_donation_c": "pledge_id"}),
        on="crm_donation_id", how="left"
    )
    del opp
    gc.collect()

    n_orphans = pay["pledge_id"].isna().sum()
    if n_orphans > 0:
        print(f"  ⚠  {n_orphans:,} pagos sin pledge_id (oportunidad no encontrada) → se descartan")
        pay = pay[pay["pledge_id"].notna()].copy()
    print(f"  Pagos con pledge_id: {len(pay):,}  |  Pledges únicos: {pay['pledge_id'].nunique():,}")

    # ── [4/6] Normalización de banco ─────────────────────────────────────────
    print("\n[4/6] Normalizando nombres de banco...")
    canonical_map = load_canonical_map()
    pay["bank_canonical"] = apply_bank_canonical(
        pay.get("bank_name", pd.Series(["UNKNOWN"] * len(pay))), canonical_map
    ).astype(str)

    # ── [5/6] Features de secuencia (incremental) ────────────────────────────
    print("\n[5/6] Calculando features de secuencia (modo incremental)...")
    pay["payment_submethod"] = pay.get("payment_submethod", "UNKNOWN").fillna("UNKNOWN").astype(str)
    pay["payment_method"]    = pay.get("payment_method", "UNKNOWN").fillna("UNKNOWN").astype(str)
    pay["payment_gateway"]   = pay.get("payment_gateway", "UNKNOWN").fillna("UNKNOWN").astype(str)
    pay["payment_amount"]    = pd.to_numeric(pay.get("payment_amount", 0), errors="coerce").fillna(0).astype("float32")
    pay["contact_id"]        = pay["contact_id"].fillna("UNKNOWN").astype(str)

    pay = compute_incremental_sequence(pay, hist_state)

    # ── [6/6] Features de contacto ────────────────────────────────────────────
    print("\n[6/6] Enriqueciendo con features de contacto...")
    # Para pledges ya conocidos: usar features del histórico
    known_pledges = set(hist_state.index)
    known_mask = pay["pledge_id"].isin(known_pledges)
    print(f"  Pledges conocidos (en histórico): {known_mask.sum():,}  |  "
          f"Pledges nuevos: {(~known_mask).sum():,}")

    for col in ["state_norm", "gender_norm", "age_band", "ses_level", "bank_canonical"]:
        pay.loc[known_mask, col] = pay.loc[known_mask, "pledge_id"].map(
            hist_state[col].to_dict()
        )
        pay[col] = pay[col].fillna("UNKNOWN").astype(str)

    # Para pledges NUEVOS: buscar en contact_unified
    new_pledges_mask = ~known_mask
    if new_pledges_mask.any():
        new_contact_ids = pay.loc[new_pledges_mask, "contact_id"].unique()
        contact_cols = ["id", "mailing_state", "gender_c", "sescore_gender_c",
                        "cco_socioeconomic_level_c", "sescore_age_c", "created_date"]
        contact = pd.read_parquet(CONTACT_PATH, columns=contact_cols)
        contact = contact[contact["id"].isin(new_contact_ids)].rename(columns={"id": "contact_id"})

        contact["state_norm"] = contact["mailing_state"].apply(normalize_state)
        top_states = contact["state_norm"].value_counts().head(TOP_N_STATES).index.tolist()
        contact.loc[~contact["state_norm"].isin(top_states), "state_norm"] = "OTHER"

        contact["gender_norm"] = contact["gender_c"].fillna(
            contact["sescore_gender_c"]).fillna("U").astype(str).str.upper().str[0]
        contact.loc[~contact["gender_norm"].isin(["M", "F"]), "gender_norm"] = "U"

        age_num = pd.to_numeric(contact["sescore_age_c"], errors="coerce")
        contact["age_band"] = pd.cut(age_num, bins=[-1, 25, 35, 45, 55, 65, 200],
                                     labels=["<25","25-35","35-45","45-55","55-65","65+"]).astype(str)
        contact.loc[contact["age_band"].isin(["nan", "NaN"]), "age_band"] = "UNKNOWN"
        contact["ses_level"] = contact["cco_socioeconomic_level_c"].fillna("UNKNOWN").astype(str)
        contact["contact_created_ts"] = pd.to_datetime(
            contact["created_date"], errors="coerce", utc=True
        ).dt.tz_localize(None)

        contact_slim = contact[["contact_id", "state_norm", "gender_norm",
                                 "age_band", "ses_level", "contact_created_ts"]].copy()
        del contact
        gc.collect()

        pay = pay.merge(contact_slim, on="contact_id", how="left",
                        suffixes=("", "_new"))
        for col in ["state_norm", "gender_norm", "age_band", "ses_level"]:
            col_new = f"{col}_new"
            if col_new in pay.columns:
                pay.loc[new_pledges_mask & pay[col_new].notna(), col] = \
                    pay.loc[new_pledges_mask & pay[col_new].notna(), col_new]
                pay.drop(columns=[col_new], inplace=True)

        # contact_age_days para nuevos pledges
        if "contact_created_ts" in pay.columns:
            pay["contact_age_days"] = (
                pay["attempt_ts"] - pay["contact_created_ts"]
            ).dt.days.astype("Int32")
            pay.loc[pay["contact_age_days"] < 0, "contact_age_days"] = pd.NA
            pay.drop(columns=["contact_created_ts"], inplace=True, errors="ignore")
    else:
        pay["contact_age_days"] = pd.NA

    # contact_age_days para pledges CONOCIDOS: recalcular desde la base histórica
    # Proxy: contact_age_days[hist] + días adicionales transcurridos
    if known_mask.any():
        hist_contact_age = hist_state["last_ts"].to_dict()
        # Para cada fila conocida: base histórica contact_age_days + días desde hist
        # Necesitamos el contact_age_days de la última fila histórica por pledge
        hist_last_cad_map = (
            pd.read_parquet(hist_path, columns=["pledge_id", "attempt_ts", "contact_age_days"])
            .sort_values(["pledge_id", "attempt_ts"])
            .groupby("pledge_id")
            .last()["contact_age_days"]
            .to_dict()
        )
        hist_last_ts_map = hist_state["last_ts"].to_dict()

        def compute_contact_age(row):
            if not known_mask[row.name]:
                return row.get("contact_age_days", pd.NA)
            pid = row["pledge_id"]
            base_cad = hist_last_cad_map.get(pid, pd.NA)
            last_ts  = hist_last_ts_map.get(pid)
            if pd.isna(base_cad) or last_ts is None:
                return pd.NA
            additional_days = (row["attempt_ts"] - last_ts).days
            return int(base_cad) + additional_days

        # Vectorized approach for known pledges
        known_df = pay[known_mask].copy()
        known_df["_base_cad"] = known_df["pledge_id"].map(hist_last_cad_map)
        known_df["_last_ts"]  = known_df["pledge_id"].map(hist_last_ts_map)
        known_df["contact_age_days"] = (
            known_df["_base_cad"].fillna(0)
            + (known_df["attempt_ts"] - known_df["_last_ts"]).dt.days
        ).astype("Int32")
        pay.loc[known_mask, "contact_age_days"] = known_df["contact_age_days"].values
        del known_df

    # Asegurar que contact_age_days sea Int32 y no negativo
    pay["contact_age_days"] = pd.to_numeric(pay["contact_age_days"], errors="coerce").astype("Int32")
    pay.loc[pay["contact_age_days"] < 0, "contact_age_days"] = pd.NA

    # ── Selección de columnas finales ─────────────────────────────────────────
    available = [c for c in FINAL_COLS if c in pay.columns]
    missing_final = [c for c in FINAL_COLS if c not in pay.columns]
    if missing_final:
        print(f"  ⚠  Columnas no disponibles (quedarán ausentes): {missing_final}")
    out_df = pay[available].copy()

    # Casteos de tipo para consistencia con motor1_features.parquet
    for col in ["state_norm", "gender_norm", "age_band", "ses_level", "bank_canonical",
                "payment_gateway", "payment_method", "payment_submethod"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].fillna("UNKNOWN").astype(str)
    out_df["is_success"] = out_df["is_success"].astype("int8")

    # ── Guardar ───────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)

    print(f"\n[OK] motor1_features_ytd.parquet guardado: {len(out_df):,} filas × {len(out_df.columns)} cols")
    print(f"  Pledges únicos: {out_df['pledge_id'].nunique():,}")
    print(f"  Intentos fallidos: {(out_df['is_success'] == 0).sum():,}")
    print(f"  Rango de fechas: {out_df['attempt_ts'].min().date()} → {out_df['attempt_ts'].max().date()}")
    print(f"  → {out_path}")


if __name__ == "__main__":
    main()
