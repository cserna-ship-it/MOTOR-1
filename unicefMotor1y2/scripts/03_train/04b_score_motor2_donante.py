#!/usr/bin/env python3
"""
04b_score_motor2_donante.py — Puntuación mensual de Motor 2 a nivel DONANTE

QUÉ CAMBIA RESPECTO A 04_score_motor2.py (pledge-level)
---------------------------------------------------------
Mismo propósito (aplicar el modelo YA ENTRENADO al snapshot del mes en curso,
sin reentrenar), pero antes de puntuar se agrega el snapshot a nivel donante.
La agregación y la imputación vienen de scripts/motor2_donante_common.py — el
MISMO módulo que usa 01b_motor2_dataset_donante.py para construir el dataset de
entrenamiento. Si el snapshot mensual se agregara o se imputara con una regla
distinta a la que vio el modelo, las predicciones saldrían sesgadas sin ningún
error visible; compartir el módulo hace imposible que las dos se desincronicen.

REQUISITO PREVIO
----------------
El snapshot mensual pledge-level (el que genera 01_motor2_dataset.py con
--months 1) DEBE incluir la columna contact_id — ya la agrega la versión actual
de ese script.

ENTRADA / SALIDA
----------------
Lee:    results/motor2_donante/motor2_donante_catboost.cbm
        results/motor2_donante/motor2_donante_calibrador.pkl
        results/motor2_donante/motor2_donante_cutoff.json   (umbral de corte)
        data/features/motor2_snapshot_<AAAA_MM_01>.parquet  (pledge-level, CON
        contact_id — output de 01_motor2_dataset.py --months 1)
Genera: results/motor2_donante/scoring_motor2_donante_<AAAA_MM_DD>.parquet
        results/motor2_donante/scoring_motor2_donante_<AAAA_MM_DD>.csv

QUÉ TRAE LA SALIDA
------------------
Las DOS cosas, no una u otra:
  prob_churn_donante  probabilidad calibrada (0–1) de que el donante cancele
                      todos sus pledges en los próximos 30 días. Sirve para
                      ORDENAR la lista y priorizar dentro de ella.
  alerta              1 si prob_churn_donante >= cutoff, 0 si no. Es la
                      DECISIÓN: quién entra a la lista de retención este mes.
El cutoff NO se recalcula acá: se lee del artefacto que dejó el entrenamiento
y se verifica que corresponda al mismo .cbm (huella sha256).
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/
from motor2_donante_common import (
    CUTOFF_FILENAME, adjuntar_contact_id, agregar_a_donante, aplicar_cutoff,
    cargar_cutoff, preparar_features_donante,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "results/motor2_donante/motor2_donante_catboost.cbm"
CALIB_PATH = ROOT / "results/motor2_donante/motor2_donante_calibrador.pkl"
CUTOFF_PATH = ROOT / "results/motor2_donante" / CUTOFF_FILENAME
STAGING = ROOT / "data/staging/stg_payment_attempts.parquet"
OUT = ROOT / "results/motor2_donante"

# Columnas de contexto para que retención entienda POR QUÉ un donante está
# arriba en la lista. No entran al modelo.
COLS_CONTEXTO = ["n_pledges_activos", "streak_max", "tenure_min",
                 "days_since_last_attempt_min", "monto_total_mensual"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Puntúa el snapshot del mes con el modelo de donante.")
    p.add_argument("--score-date", type=str, default=None,
                   help="Fecha de ejecución 'hoy' (YYYY-MM-DD). Default: fecha del sistema.")
    p.add_argument("--dataset", type=str, required=True,
                   help="Parquet pledge-level del snapshot del mes (CON contact_id), "
                        "output de 01_motor2_dataset.py --months 1.")
    return p.parse_args()


def main():
    args = parse_args()
    SCORE_DATE = (pd.Timestamp(args.score_date) if args.score_date
                  else pd.Timestamp.today().normalize())
    SLUG_RUN = SCORE_DATE.strftime("%Y_%m_%d")

    for path, que in [(MODEL_PATH, "modelo"), (CALIB_PATH, "calibrador")]:
        if not path.exists():
            raise SystemExit(
                f"\n[ERROR] Falta el {que} de Motor 2 donante: {path}\n"
                f"        Entrena primero: "
                f"python3 scripts/03_train/02b_train_motor2_donante.py\n")

    print("[1/4] Cargando modelo, calibrador y cutoff de donante...")
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    with open(CALIB_PATH, "rb") as f:
        calibrador = pickle.load(f)
    features = list(model.feature_names_)
    print(f"  Features del modelo: {len(features)}")
    cutoff_info = cargar_cutoff(CUTOFF_PATH, MODEL_PATH)
    cutoff = float(cutoff_info["cutoff"])
    print(f"  Cutoff ({cutoff_info.get('metodo')}): {cutoff:.4f}  (calculado en {cutoff_info.get('calculado_en')}; "
          f"sens {cutoff_info['sensibilidad']:.1%} / espec {cutoff_info['especificidad']:.1%})")

    dataset = Path(args.dataset)
    if not dataset.exists():
        raise SystemExit(f"\n[ERROR] No existe el snapshot pledge-level: {dataset}\n")
    print(f"[2/4] Cargando snapshot pledge-level: {dataset.name}")
    df = pd.read_parquet(dataset)
    if "contact_id" not in df.columns:
        raise SystemExit(
            "\n[ERROR] El snapshot no tiene la columna contact_id.\n"
            "        Regenéralo con la versión actual de 01_motor2_dataset.py.\n")
    print(f"  Filas pledge-level: {len(df):,}")
    print(f"  Snapshot(s): "
          f"{', '.join(sorted(df['snapshot_ts'].dt.strftime('%Y-%m-%d').unique()))}")

    print("[3/4] Agregando a nivel donante (mismas reglas que el entrenamiento)...")
    df = adjuntar_contact_id(df, STAGING)
    donante_df = agregar_a_donante(df, con_label=False)
    donante_df = preparar_features_donante(donante_df)

    faltan = [c for c in features if c not in donante_df.columns]
    if faltan:
        raise SystemExit(
            f"\n[ERROR] Al snapshot agregado le faltan features que el modelo espera: {faltan}\n"
            f"        La agregación se desincronizó de 02b_train_motor2_donante.py.\n")
    print(f"  Donantes a puntuar: {len(donante_df):,}")

    print("[4/4] Puntuando...")
    prob_cruda = model.predict_proba(donante_df[features])[:, 1]
    prob = calibrador.transform(prob_cruda)

    out = donante_df[["contact_id", "snapshot_ts"]].copy()
    out["prob_churn_donante"] = prob.astype("float32")
    out["prob_churn_donante_cruda"] = prob_cruda.astype("float32")
    out["alerta"] = aplicar_cutoff(prob, cutoff)
    for c in COLS_CONTEXTO:
        if c in donante_df.columns:
            out[c] = donante_df[c].values
    out = out.sort_values("prob_churn_donante", ascending=False).reset_index(drop=True)

    pq_path = OUT / f"scoring_motor2_donante_{SLUG_RUN}.parquet"
    csv_path = OUT / f"scoring_motor2_donante_{SLUG_RUN}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(pq_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"\n  Donantes puntuados: {len(out):,}   |   "
          f"P2 medio: {out['prob_churn_donante'].mean():.4f}")
    n_alerta = int(out["alerta"].sum())
    print(f"  En alerta (prob >= {cutoff:.4f}): {n_alerta:,} donantes "
          f"({n_alerta / max(len(out), 1):.1%})  — referencia en test: "
          f"{cutoff_info['pct_donantes_en_alerta']:.1%}")
    print(f"  [OK] {csv_path}")
    print(f"  [OK] {pq_path}")


if __name__ == "__main__":
    main()
