#!/usr/bin/env python3
"""
01b_experimento_motor1.py — Efecto de auto_class_weights en el Motor 1

NOTA: este NO es un script del pipeline (no lo llama run_all/run_monthly). Es un
experimento reproducible que documenta el efecto real de `auto_class_weights`
sobre el Motor 1, para dejar evidencia numérica de una decisión que hasta ahora
solo estaba justificada de forma cualitativa en el docstring de
`scripts/03_train/01_train_motor1.py` ("en pruebas el AUC resultó levemente
inferior que sin balanceo").

Ejecutar desde la raíz del entregable:
    .venv/bin/python docs/01b_experimento_motor1.py

NO toca los artefactos de producción (results/motor1/). Escribe todo en
results/motor1_experimentos/.

DISEÑO
------
Se reutilizan `load_slice`, `FEATURES` y `FEAT_CAT` del script de producción vía
importlib, de modo que el preprocesamiento, el split temporal, el orden canónico
y el holdout de validación son BIT A BIT los mismos que en producción. La única
variable que cambia entre configuraciones es `auto_class_weights`:

    None  ·  Balanced (producción v8)  ·  SqrtBalanced

Se reporta además el peso efectivo que CatBoost asigna a cada clase, calculado
a partir de los conteos reales del train, para dimensionar la intervención.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score, recall_score,
    precision_score,
)
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/motor1_experimentos"
OUT.mkdir(parents=True, exist_ok=True)

# Importar el script de producción como módulo (su nombre empieza por dígito,
# así que no se puede usar un import normal).
_spec = importlib.util.spec_from_file_location(
    "train_motor1", ROOT / "scripts/03_train/01_train_motor1.py")
m1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m1)

FEATURES = m1.FEATURES
FEAT_CAT = m1.FEAT_CAT

CONFIGS = [
    {"nombre": "M1_sin_pesos",     "weights": None},
    {"nombre": "M1_balanced_prod", "weights": "Balanced"},      # config v8 en producción
    {"nombre": "M1_sqrt_balanced", "weights": "SqrtBalanced"},
]


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error. 0 = calibración perfecta."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    err = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(probs[mask].mean() - labels[mask].mean())
    return err


def run_config(cfg, train, test, tr_idx, val_idx):
    """Entrena una configuración con el pipeline idéntico al de producción."""
    kw = dict(
        iterations=180, depth=6, learning_rate=0.10,
        loss_function="Logloss", eval_metric="AUC",
        cat_features=FEAT_CAT, random_seed=42,
        verbose=False, thread_count=-1,
        od_type="Iter", od_wait=15,
    )
    if cfg["weights"] is not None:
        kw["auto_class_weights"] = cfg["weights"]

    model = CatBoostClassifier(**kw)
    model.fit(
        train.loc[tr_idx, FEATURES], train.loc[tr_idx, "is_success"],
        eval_set=(train.loc[val_idx, FEATURES], train.loc[val_idx, "is_success"]),
        use_best_model=True,
    )

    # Calibración isotónica sobre el val holdout (idéntico a producción).
    prob_val = model.predict_proba(train.loc[val_idx, FEATURES])[:, 1]
    y_val = train.loc[val_idx, "is_success"].values
    cal = IsotonicRegression(out_of_bounds="clip").fit(prob_val, y_val)

    y_crudo = model.predict_proba(test[FEATURES])[:, 1]
    y_cal = cal.transform(y_crudo)
    y_true = test["is_success"].values

    # Métricas @ threshold 0.5 sobre la probabilidad CALIBRADA.
    pred05 = (y_cal >= 0.5).astype(int)

    # La clase accionable del negocio es el FALLO (is_success == 0): son los
    # cobros que se pueden reintentar. Se mide explícitamente sobre esa clase.
    y_fallo = 1 - y_true
    pred_fallo = 1 - pred05

    # Métrica operacional: top 20% por riesgo de fallo (1 - prob_exito).
    riesgo = 1.0 - y_cal
    thr20 = np.quantile(riesgo, 0.80)
    pred_top20 = (riesgo >= thr20).astype(int)
    recall20 = (pred_top20 & y_fallo).sum() / max(y_fallo.sum(), 1)
    prec20 = (pred_top20 & y_fallo).sum() / max(pred_top20.sum(), 1)

    return {
        "nombre": cfg["nombre"],
        "weights": str(cfg["weights"]),
        "best_iteration": int(model.get_best_iteration()),
        "auc": float(roc_auc_score(y_true, y_cal)),
        "pr_auc": float(average_precision_score(y_true, y_cal)),
        "brier_crudo": float(brier_score_loss(y_true, y_crudo)),
        "brier_calibrado": float(brier_score_loss(y_true, y_cal)),
        "ece_test_antes": float(ece(y_crudo, y_true)),
        "ece_test_despues": float(ece(y_cal, y_true)),
        "media_prob_cruda": float(y_crudo.mean()),
        "media_prob_calibrada": float(y_cal.mean()),
        "tasa_exito_test_real": float(y_true.mean()),
        # Clase 1 = éxito
        "recall_exito_@0.5": float(recall_score(y_true, pred05)),
        "precision_exito_@0.5": float(precision_score(y_true, pred05)),
        # Clase 0 = fallo (la accionable)
        "recall_fallo_@0.5": float(recall_score(y_fallo, pred_fallo)),
        "precision_fallo_@0.5": float(precision_score(y_fallo, pred_fallo)),
        "recall_fallo_top20": float(recall20),
        "precision_fallo_top20": float(prec20),
    }


def main():
    CUTOFF = m1.get_cutoff_date()
    TEST_YEAR = CUTOFF.year
    TRAIN_END = TEST_YEAR - 1
    TRAIN_START = 2022

    print(f"[1/3] Cargando train (años {TRAIN_START}-{TRAIN_END})...")
    train = m1.load_slice(TRAIN_START, TRAIN_END, sample_frac=1.00)
    print(f"  Train: {len(train):,}")
    print(f"[2/3] Cargando test (año {TEST_YEAR})...")
    test = m1.load_slice(TEST_YEAR, TEST_YEAR)
    print(f"  Test: {len(test):,}")

    # === Diagnóstico del desbalance real y de los pesos que aplicaría CatBoost ===
    # CatBoost: w_clase = max(conteos) / conteo_clase   (auto_class_weights="Balanced")
    #           w_clase = sqrt(max(conteos) / conteo_clase)  ("SqrtBalanced")
    n_exito = int(train["is_success"].sum())
    n_fallo = int(len(train) - n_exito)
    n_max = max(n_exito, n_fallo)
    pesos = {
        "n_exito_train": n_exito,
        "n_fallo_train": n_fallo,
        "tasa_exito_train": float(train["is_success"].mean()),
        "clase_minoritaria": "exito (is_success=1)" if n_exito < n_fallo else "fallo (is_success=0)",
        "ratio_mayoritaria_sobre_minoritaria": float(n_max / min(n_exito, n_fallo)),
        "peso_Balanced_exito": float(n_max / n_exito),
        "peso_Balanced_fallo": float(n_max / n_fallo),
        "peso_SqrtBalanced_exito": float(np.sqrt(n_max / n_exito)),
        "peso_SqrtBalanced_fallo": float(np.sqrt(n_max / n_fallo)),
    }
    print("\n  === Desbalance real del target (train) ===")
    print(f"  éxitos: {n_exito:,}  |  fallos: {n_fallo:,}  |  tasa éxito: {pesos['tasa_exito_train']:.4f}")
    print(f"  clase MINORITARIA: {pesos['clase_minoritaria']}")
    print(f"  ratio mayoritaria/minoritaria: {pesos['ratio_mayoritaria_sobre_minoritaria']:.3f}")
    print(f"  pesos con Balanced     → éxito={pesos['peso_Balanced_exito']:.4f}  fallo={pesos['peso_Balanced_fallo']:.4f}")
    print(f"  pesos con SqrtBalanced → éxito={pesos['peso_SqrtBalanced_exito']:.4f}  fallo={pesos['peso_SqrtBalanced_fallo']:.4f}")

    val_size = int(len(train) * 0.1)
    tr_idx = train.iloc[:-val_size].index
    val_idx = train.iloc[-val_size:].index

    print("\n[3/3] Entrenando configuraciones...")
    resultados = []
    for cfg in CONFIGS:
        print(f"\n=== {cfg['nombre']} (auto_class_weights={cfg['weights']}) ===")
        r = run_config(cfg, train, test, tr_idx, val_idx)
        resultados.append(r)
        print(f"  best_iter={r['best_iteration']:>3}  AUC={r['auc']:.4f}  PR-AUC={r['pr_auc']:.4f}  "
              f"ECE {r['ece_test_antes']:.4f}->{r['ece_test_despues']:.4f}  "
              f"Brier_cal={r['brier_calibrado']:.4f}")
        print(f"  clase FALLO (accionable): recall@0.5={r['recall_fallo_@0.5']:.4f}  "
              f"precision@0.5={r['precision_fallo_@0.5']:.4f}  "
              f"recall@top20={r['recall_fallo_top20']:.4f}")

    with open(OUT / "comparacion.json", "w") as f:
        json.dump({"desbalance_y_pesos": pesos, "resultados": resultados}, f, indent=2)

    print("\n\n" + "=" * 118)
    print(f"{'config':<20}{'weights':<15}{'best_it':>8}{'AUC':>9}{'PR-AUC':>9}"
          f"{'ECE_post':>10}{'Brier':>9}{'Rec_fallo':>11}{'Prec_fallo':>12}{'R_fallo@20':>12}")
    print("=" * 118)
    for r in resultados:
        print(f"{r['nombre']:<20}{r['weights']:<15}{r['best_iteration']:>8}"
              f"{r['auc']:>9.4f}{r['pr_auc']:>9.4f}{r['ece_test_despues']:>10.4f}"
              f"{r['brier_calibrado']:>9.4f}{r['recall_fallo_@0.5']:>11.4f}"
              f"{r['precision_fallo_@0.5']:>12.4f}{r['recall_fallo_top20']:>12.4f}")
    print("=" * 118)
    print(f"\nSalida: {OUT / 'comparacion.json'}")


if __name__ == "__main__":
    main()
