#!/usr/bin/env python3
"""
06_experimento_calibracion_motor2.py — ¿Por qué "90%" no significa 90%?

EL PROBLEMA QUE INVESTIGA ESTE EXPERIMENTO
-------------------------------------------
El Motor 2 ordena muy bien (AUC 0,89) pero el NÚMERO que entrega no se puede
leer literal. Medido sobre el test 2025 del modelo de donante:

    el modelo dice 90%  →  se van de verdad 56%

Negocio necesita poder leer `prob_churn` como "de cada 100 así, se van N".
Hoy no puede. Este experimento aísla las dos causas candidatas y mide cuál
de las dos (o las dos juntas) lo arregla.

CAUSA 1 — `boost_from_average=False`
-------------------------------------
Es el default de CatBoost para Logloss. El modelo arranca prediciendo 0,5 (una
moneda) en vez de la tasa base (~3,5%), y va bajando árbol por árbol. Pero el
early stopping mira AUC — o sea el ORDEN — y corta apenas el orden deja de
mejorar, que pasa mucho antes de que el número llegue a su valor correcto.

    pledge-level: para en 60 árboles → media cruda 0,052 (real 0,035)
    donante:      para en 21 árboles → media cruda 0,139 (real 0,033)

CAUSA 2 — el validation set no es representativo
-------------------------------------------------
Se toma como el último 10% del train ORDENADO POR `pledge_id` / `contact_id`.
Los IDs de Salesforce son casi cronológicos (se asignan en orden de creación),
así que ese corte "por posición" resulta ser un corte TEMPORAL encubierto: se
queda con los donantes más nuevos, que se van más.

    val actual (posicional): tasa de churn 0,0590
    train restante:                        0,0426
    test 2025 (la realidad):               0,0348

El calibrador isotónico se ajusta contra ese val pesimista y hereda el sesgo:
por eso sobreestima justo en el tramo alto, que es el que usa negocio.

QUÉ COMPARA
-----------
    A  boost_from_average=False, split posicional   ← configuración actual
    B  boost_from_average=True,  split posicional
    C  boost_from_average=False, split temporal
    D  boost_from_average=True,  split temporal     ← candidato

Split temporal = los últimos N meses del train (default 2) como validación, en
vez del último 10% por posición.

LA MÉTRICA QUE DECIDE
---------------------
No es el AUC (el orden ya está bien y no debería cambiar). Es el **error de
calibración en el tramo alto**: la diferencia entre lo que el modelo promete y
lo que ocurre, dentro del top 1% / 5% / 10% — que es exactamente la población a
la que retención va a llamar.

USO
---
    python3 docs/06_experimento_calibracion_motor2.py                # pledge
    python3 docs/06_experimento_calibracion_motor2.py --nivel donante
    python3 docs/06_experimento_calibracion_motor2.py --nivel ambos

No escribe nada en results/ ni toca producción: solo imprime la comparación.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from motor2_donante_common import FEATURES_DONANTE, preparar_features_donante

CUTOFF = pd.Timestamp("2025-12-31")
TEST_YEAR, TRAIN_END = CUTOFF.year, CUTOFF.year - 1

FEAT_NUM_PLEDGE = ["tenure_days", "n_intentos_30d", "n_fallos_30d", "streak_actual",
                   "success_rate_hist", "cum_intentos", "cum_exitos", "cum_fallos",
                   "days_since_last_attempt", "last_payment_amount", "last_attempt_success"]
FEAT_CAT_PLEDGE = ["last_payment_method", "last_bank_name", "last_gateway"]


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum():
            err += m.sum() / len(probs) * abs(probs[m].mean() - labels[m].mean())
    return err


def error_tramo_alto(prob, y, pct):
    """
    Lo que le importa a negocio: dentro del top `pct` (a quien se va a llamar),
    ¿cuánto se aleja lo que el modelo promete de lo que de verdad pasa?
    """
    k = max(int(len(prob) * pct), 1)
    idx = np.argsort(-prob)[:k]
    return prob[idx].mean(), y[idx].mean()


def cargar(nivel):
    if nivel == "pledge":
        df = pd.read_parquet(ROOT / "data/features/motor2_pledge_month.parquet")
        for c in FEAT_CAT_PLEDGE:
            df[c] = df[c].astype(str).fillna("Unknown")
        df["last_payment_amount"] = df["last_payment_amount"].fillna(0).astype("float32")
        df["days_since_last_attempt"] = df["days_since_last_attempt"].fillna(999).astype("float32")
        df["last_attempt_success"] = df["last_attempt_success"].fillna(-1).astype("int8")
        df = df.sort_values(["pledge_id", "snapshot_ts"]).reset_index(drop=True)
        return df, FEAT_NUM_PLEDGE + FEAT_CAT_PLEDGE, FEAT_CAT_PLEDGE
    df = pd.read_parquet(ROOT / "data/features/motor2_donante_month.parquet")
    df = preparar_features_donante(df)
    df = df.sort_values(["contact_id", "snapshot_ts"]).reset_index(drop=True)
    return df, FEATURES_DONANTE, []


def split_val(train, modo, meses_val):
    """Devuelve (tr_idx, val_idx) según el modo de partición."""
    if modo == "posicional":
        n = int(len(train) * 0.1)
        return train.iloc[:-n].index, train.iloc[-n:].index
    corte = sorted(train["snapshot_ts"].unique())[-meses_val]
    es_val = train["snapshot_ts"] >= corte
    return train.index[~es_val], train.index[es_val]


def correr(nivel, meses_val):
    print("\n" + "=" * 78)
    print(f"  NIVEL: {nivel.upper()}")
    print("=" * 78)
    df, features, cat = cargar(nivel)
    train = df[df.snapshot_ts.dt.year <= TRAIN_END]
    test = df[df.snapshot_ts.dt.year == TEST_YEAR]
    y_test = test["label_churn_30d"].values
    base = y_test.mean()
    print(f"  train={len(train):,}  test={len(test):,}  tasa churn test={base:.4f}")

    filas = []
    for etiqueta, bfa, modo in [("A actual   (bfa=False, posicional)", False, "posicional"),
                                ("B          (bfa=True,  posicional)", True, "posicional"),
                                ("C          (bfa=False, temporal)", False, "temporal"),
                                ("D candidato(bfa=True,  temporal)", True, "temporal")]:
        tr_idx, val_idx = split_val(train, modo, meses_val)
        tasa_val = train.loc[val_idx, "label_churn_30d"].mean()

        m = CatBoostClassifier(
            iterations=800, depth=6, learning_rate=0.03,
            loss_function="Logloss", eval_metric="AUC",
            cat_features=cat, random_seed=42, verbose=0, thread_count=-1,
            od_type="Iter", od_wait=15, boost_from_average=bfa,
        )
        m.fit(train.loc[tr_idx, features], train.loc[tr_idx, "label_churn_30d"],
              eval_set=(train.loc[val_idx, features], train.loc[val_idx, "label_churn_30d"]),
              use_best_model=True)

        pv = m.predict_proba(train.loc[val_idx, features])[:, 1]
        cal = IsotonicRegression(out_of_bounds="clip").fit(
            pv, train.loc[val_idx, "label_churn_30d"].values)
        crudo = m.predict_proba(test[features])[:, 1]
        calib = cal.transform(crudo)

        fila = {
            "config": etiqueta,
            "n_val%": f"{100*len(val_idx)/len(train):.0f}%",
            "tasa_val": f"{tasa_val:.4f}",
            "arb": m.get_best_iteration(),
            "AUC": roc_auc_score(y_test, calib),
            "PR-AUC": average_precision_score(y_test, calib),
            "media_cruda": crudo.mean(),
            "ECE_crudo": ece(crudo, y_test),
            "ECE_calib": ece(calib, y_test),
        }
        for pct, nom in [(0.01, "top1%"), (0.05, "top5%"), (0.10, "top10%")]:
            promete, ocurre = error_tramo_alto(calib, y_test, pct)
            fila[f"{nom} dice"] = promete
            fila[f"{nom} real"] = ocurre
            fila[f"{nom} err"] = promete - ocurre
        filas.append(fila)

    r = pd.DataFrame(filas).set_index("config")
    print(f"\n  --- Ranking y calibración global (tasa base real: {base:.4f}) ---")
    print(r[["n_val%", "tasa_val", "arb", "AUC", "PR-AUC", "media_cruda",
             "ECE_crudo", "ECE_calib"]].round(4).to_string())
    print(f"\n  --- LO QUE DECIDE: ¿el número se puede leer literal en el tramo alto? ---")
    print("      (dice = lo que promete el modelo · real = lo que ocurre · err = sobreestimación)")
    print(r[[c for c in r.columns if c.startswith("top")]].round(3).to_string())

    err = r[[c for c in r.columns if c.endswith("err")]].abs().mean(axis=1)
    print(f"\n  Error medio absoluto en los tres tramos altos:")
    for k, v in err.sort_values().items():
        print(f"    {v:.4f}   {k}")
    print(f"\n  → GANA: {err.idxmin()}")
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nivel", choices=["pledge", "donante", "ambos"], default="pledge")
    p.add_argument("--meses-val", type=int, default=2,
                   help="Meses finales del train usados como validación en el split temporal.")
    a = p.parse_args()
    niveles = ["pledge", "donante"] if a.nivel == "ambos" else [a.nivel]
    for n in niveles:
        correr(n, a.meses_val)


if __name__ == "__main__":
    main()
