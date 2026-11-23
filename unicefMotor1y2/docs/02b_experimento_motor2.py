#!/usr/bin/env python3
"""
02b_experimento_motor2.py — Comparación de alternativas de configuración (Motor 2)

NOTA: este NO es un script del pipeline (no lo llama run_all/run_monthly). Es el
experimento de un solo uso que justificó la config de producción del Motor 2
(quitar auto_class_weights="Balanced"). Se conserva en docs/ como evidencia
reproducible. Ejecutar desde la raíz del entregable:
    .venv/bin/python docs/02b_experimento_motor2.py

NO toca los artefactos de producción (results/motor2/). Escribe todo en
results/motor2_experimentos/. Aísla dos ejes de decisión:

  EJE 1 — Pesos de clase:  None  vs  Balanced  vs  SqrtBalanced
  EJE 2 — Learning rate:   lr=0.10/iter=150  vs  lr=0.03/iter=800

Para cada config aplica EXACTAMENTE el mismo pipeline que producción
(mismo split temporal, mismo val holdout, misma calibración isotónica,
mismo seed) para que la única diferencia sea la config comparada.

Objetivo: responder si quitar auto_class_weights="Balanced" y/o bajar el
learning rate mejora AUC / calibración / y corrige el early-stopping en 4 árboles.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from catboost import CatBoostClassifier

# docs/ está un nivel bajo la raíz del entregable → parents[1].
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/features/motor2_pledge_month.parquet"
OUT = ROOT / "results/motor2_experimentos"
OUT.mkdir(parents=True, exist_ok=True)

SORT_KEYS = ["pledge_id", "snapshot_ts"]
feat_num = ["tenure_days", "n_intentos_30d", "n_fallos_30d", "streak_actual",
            "success_rate_hist", "cum_intentos", "cum_exitos", "cum_fallos",
            "days_since_last_attempt", "last_payment_amount", "last_attempt_success"]
feat_cat = ["last_payment_method", "last_bank_name", "last_gateway"]
features = feat_num + feat_cat


def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs); err = 0.0
    for i in range(n_bins):
        m = (probs >= bins[i]) & (probs < bins[i + 1])
        if m.sum() == 0:
            continue
        err += m.sum() / total * abs(probs[m].mean() - labels[m].mean())
    return err


def prep(df):
    for c in ["last_payment_method", "last_bank_name", "last_gateway"]:
        df[c] = df[c].astype(str).fillna("Unknown")
    df["last_payment_amount"] = df["last_payment_amount"].fillna(0).astype("float32")
    df["days_since_last_attempt"] = df["days_since_last_attempt"].fillna(999).astype("float32")
    df["last_attempt_success"] = df["last_attempt_success"].fillna(-1).astype("int8")
    return df.sort_values(SORT_KEYS).reset_index(drop=True)


# Configuraciones a comparar. auto_class_weights=None → sin pesos.
CONFIGS = [
    {"nombre": "A_v2_actual",        "weights": "Balanced",     "lr": 0.10, "iters": 150},
    {"nombre": "B_sin_pesos",        "weights": None,           "lr": 0.10, "iters": 150},
    {"nombre": "C_sqrt_balanced",    "weights": "SqrtBalanced", "lr": 0.10, "iters": 150},
    {"nombre": "D_sin_pesos_lr003",  "weights": None,           "lr": 0.03, "iters": 800},
    {"nombre": "E_balanced_lr003",   "weights": "Balanced",     "lr": 0.03, "iters": 800},
    {"nombre": "F_sqrt_lr003",       "weights": "SqrtBalanced", "lr": 0.03, "iters": 800},
]


def run_config(cfg, train, test, tr_idx, val_idx):
    kw = dict(
        iterations=cfg["iters"], depth=6, learning_rate=cfg["lr"],
        loss_function="Logloss", eval_metric="AUC", cat_features=feat_cat,
        random_seed=42, verbose=False, thread_count=-1,
        od_type="Iter", od_wait=15,
    )
    if cfg["weights"] is not None:
        kw["auto_class_weights"] = cfg["weights"]
    model = CatBoostClassifier(**kw)
    model.fit(
        train.loc[tr_idx, features], train.loc[tr_idx, "label_churn_30d"],
        eval_set=(train.loc[val_idx, features], train.loc[val_idx, "label_churn_30d"]),
        use_best_model=True,
    )
    best_iter = model.get_best_iteration()

    # Calibración isotónica sobre el val holdout (idéntico a producción)
    prob_val = model.predict_proba(train.loc[val_idx, features])[:, 1]
    y_val = train.loc[val_idx, "label_churn_30d"].values
    cal = IsotonicRegression(out_of_bounds="clip").fit(prob_val, y_val)

    y_crudo = model.predict_proba(test[features])[:, 1]
    y_cal = cal.transform(y_crudo)
    y_true = test["label_churn_30d"].values

    thr = np.quantile(y_cal, 0.80)
    pred20 = (y_cal >= thr).astype(int)
    recall20 = (pred20 & y_true).sum() / max(y_true.sum(), 1)
    prec20 = (pred20 & y_true).sum() / max(pred20.sum(), 1)

    return {
        "nombre": cfg["nombre"],
        "weights": str(cfg["weights"]),
        "lr": cfg["lr"], "iters_max": cfg["iters"],
        "best_iteration": int(best_iter),
        "auc": float(roc_auc_score(y_true, y_cal)),
        "pr_auc": float(average_precision_score(y_true, y_cal)),
        "brier_crudo": float(brier_score_loss(y_true, y_crudo)),
        "brier_calibrado": float(brier_score_loss(y_true, y_cal)),
        "ece_test_antes": float(ece(y_crudo, y_true)),
        "ece_test_despues": float(ece(y_cal, y_true)),
        "media_prob_cruda": float(y_crudo.mean()),
        "media_prob_calibrada": float(y_cal.mean()),
        "recall_top20": float(recall20),
        "precision_top20": float(prec20),
    }


def main():
    print("Cargando y preparando dataset...")
    df = prep(pd.read_parquet(DATASET))
    train = df[df["snapshot_ts"].dt.year < 2025].copy()
    test = df[df["snapshot_ts"].dt.year == 2025].copy()
    val_size = int(len(train) * 0.1)
    tr_idx = train.iloc[:-val_size].index
    val_idx = train.iloc[-val_size:].index
    print(f"  train={len(train):,}  test={len(test):,}  "
          f"tasa_churn_test={test['label_churn_30d'].mean():.4f}")

    resultados = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['nombre']} (weights={cfg['weights']}, lr={cfg['lr']}, iters={cfg['iters']}) ===")
        r = run_config(cfg, train, test, tr_idx, val_idx)
        resultados.append(r)
        print(f"  best_iter={r['best_iteration']:>3}  AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"ECE {r['ece_test_antes']:.3f}->{r['ece_test_despues']:.4f}  "
              f"Brier_cal={r['brier_calibrado']:.4f}  R@20={r['recall_top20']:.3f}  P@20={r['precision_top20']:.3f}")

    with open(OUT / "comparacion.json", "w") as f:
        json.dump(resultados, f, indent=2)

    # Tabla resumen
    print("\n\n" + "=" * 110)
    print(f"{'config':<18}{'weights':<14}{'lr':>5}{'best_it':>8}{'AUC':>9}{'PR-AUC':>9}"
          f"{'ECE_post':>10}{'Brier':>9}{'R@20':>8}{'P@20':>8}")
    print("=" * 110)
    for r in resultados:
        print(f"{r['nombre']:<18}{r['weights']:<14}{r['lr']:>5}{r['best_iteration']:>8}"
              f"{r['auc']:>9.4f}{r['pr_auc']:>9.4f}{r['ece_test_despues']:>10.4f}"
              f"{r['brier_calibrado']:>9.4f}{r['recall_top20']:>8.3f}{r['precision_top20']:>8.3f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
