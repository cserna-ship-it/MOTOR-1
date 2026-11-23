#!/usr/bin/env python3
"""
02_train_motor2.py — Entrenamiento Motor 2 (churn 30 días, v3)

FECHA DE CORTE CONFIGURABLE (v9)
--------------------------------
El split temporal ya NO está hard-codeado: se deriva de la fecha de corte,
igual que en 01_train_motor1.py.

    TEST_YEAR = CUTOFF.year          → año que se evalúa / puntúa
    TRAIN     = snapshots de años <= TEST_YEAR - 1

    python3 scripts/03_train/02_train_motor2.py --cutoff-date 2026-01-03
    CUTOFF_DATE=2026-01-03 bash scripts/run_all.sh

Con el default (2025-12-31) el comportamiento es idéntico al de v8:
train = snapshots 2024, test = snapshots 2025.

El script exige que `motor2_pledge_month.parquet` haya sido generado con la
MISMA fecha de corte (compara el último snapshot del dataset contra el mes de
corte) y aborta si no coinciden. Esto evita el error silencioso de reentrenar
sobre features de un corte anterior.

DOS MODOS DE OPERACIÓN
----------------------
1. EVALUACIÓN (corte a fin de año, ej. 2025-12-31): el test tiene 12 meses de
   snapshots con label observado → se reportan AUC, PR-AUC, Brier, ECE y
   baselines, como siempre.

2. SCORING (corte en el mes en curso, ej. 2026-01-03): el test es un único
   snapshot (2026-01-01) cuya ventana de 30 días AÚN NO CERRÓ. El script
   entrena, calibra y escribe las predicciones, pero marca las métricas de
   test como censuradas (o las omite si no hay ambas clases). Es el caso de
   producción: se predice, no se evalúa.

La censura se detecta por snapshot: un label es observable solo si
`snapshot_ts + 30d <= CUTOFF`.

CAMBIO CLAVE v2 → v3
--------------------
Se ELIMINA `auto_class_weights="Balanced"` y se baja el learning rate.
El experimento docs/02b_experimento_motor2.py demostró que el balanceo era
CONTRAPRODUCENTE en este dataset:

  - Con Balanced el early stopping cortaba en ~3-4 árboles (val saturaba).
    Sin pesos el modelo entrena ~55-60 árboles con normalidad.
  - Quitar Balanced sube AUC 0.878 → 0.896 y PR-AUC 0.281 → 0.367 (+30%).
    El calibrador NO puede recuperar esto: la calibración es monótona y no
    cambia el ranking.
  - Sin Balanced las probabilidades crudas ya salen casi calibradas
    (ECE crudo 0.39 → 0.010): el calibrador pasa de "rescate" a "ajuste fino".

Config v3 (ganadora del experimento, opción D):
  - SIN auto_class_weights (logloss estándar; el calibrador lleva las
    probabilidades a la tasa real sin sacrificar ranking).
  - learning_rate = 0.03, iterations = 800 (early stopping decide el corte).
    lr más baja = modelo más robusto y menos dependiente de un corte temprano.

El resto del pipeline (split temporal, validación anti-leakage, re-orden
canónico, calibración isotónica, baselines, métricas) es idéntico a v2.

--- (documentación heredada de v2) ---

PROPÓSITO
---------
Entrenar el modelo de PROBABILIDAD DE CHURN A 30 DÍAS (P2) con CatBoost.
Para cada (pledge, snapshot mensual) predice la probabilidad de que el
donante cancele su pledge en los 30 días siguientes al snapshot.

VARIABLE OBJETIVO
-----------------
`label_churn_30d`: 1 si el pledge cancela en (snapshot, snapshot+30d], 0 si no.

SPLIT TEMPORAL OUT-OF-TIME
--------------------------
- Train: snapshots de años <= CUTOFF.year - 1   (default: 2024)
- Test:  snapshots del año CUTOFF.year          (default: 2025)

ANTI-LEAKAGE
------------
El anti-leakage (merge_asof con allow_exact_matches=False) se garantiza aguas
arriba (01_motor2_dataset.py). Este script RE-ORDENA por (pledge_id,
snapshot_ts) y VALIDA la integridad temporal antes de entrenar.

CALIBRACIÓN ISOTÓNICA
---------------------
Tras entrenar, se ajusta un IsotonicRegression sobre el validation set de
early stopping. Sin Balanced el sesgo crudo es pequeño, pero se mantiene el
calibrador porque es barato, entrega la columna `prob_churn` lista para
producción y protege ante drift interanual.

ENTRADA / SALIDA
----------------
Lee:    data/features/motor2_pledge_month.parquet
Genera: results/motor2/
            motor2_catboost.cbm              → modelo serializado
            motor2_calibrador.pkl            → calibrador isotónico
            resultados_motor2.json           → métricas
            feature_importance_motor2.csv
            test_predictions_motor2.parquet  → predicciones (insumo tabla maestra)
            test_predictions_motor2.csv      → mismas predicciones en CSV (para el equipo)
"""
import pickle
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    classification_report
)
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/ al path
from config import (get_cutoff_date, imprimir_confiabilidad,
                    split_validacion_temporal, tabla_confiabilidad)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/features/motor2_pledge_month.parquet"
OUT = ROOT / "results/motor2"
OUT.mkdir(parents=True, exist_ok=True)

# Clave de orden canónica.
SORT_KEYS = ["pledge_id", "snapshot_ts"]


def validate_temporal_integrity(df, nombre):
    """
    Blindaje anti-leakage DENTRO de este script. Comprueba tres invariantes
    sobre el dataset pledge × snapshot_mes:

    1. `days_since_last_attempt` nunca es negativo (excluyendo el sentinel 999).
    2. Los conteos acumulados (cum_intentos, cum_exitos, cum_fallos) son
       MONÓTONOS NO DECRECIENTES dentro de cada pledge ordenado por snapshot_ts.
    3. `tenure_days` >= 0 para todos los registros.

    Aborta con AssertionError si alguna invariante no se cumple.
    Requiere que `df` esté ordenado por (pledge_id, snapshot_ts).
    """
    mask_real = df["days_since_last_attempt"] < 999
    neg = int((df.loc[mask_real, "days_since_last_attempt"] < 0).sum())
    assert neg == 0, (
        f"[{nombre}] {neg} filas con days_since_last_attempt negativo → "
        "el merge_asof en 01_motor2_dataset.py no está excluyendo correctamente "
        "eventos futuros (revisar allow_exact_matches=False).")

    df_sorted = df.sort_values(SORT_KEYS)
    for col in ["cum_intentos", "cum_exitos", "cum_fallos"]:
        dec = int((df_sorted.groupby("pledge_id", sort=False)[col].diff() < 0).sum())
        assert dec == 0, (
            f"[{nombre}] {dec} filas donde {col} decrece dentro del pledge → "
            f"orden temporal roto (posible leakage en conteos acumulados).")

    neg_tenure = int((df["tenure_days"] < 0).sum())
    assert neg_tenure == 0, (
        f"[{nombre}] {neg_tenure} filas con tenure_days negativo → "
        "snapshot anterior a effective_start del pledge "
        "(filtro de snapshots inválidos en 01_motor2_dataset.py roto).")

    print(f"  [OK anti-leakage] {nombre}: days≥0, conteos monótonos, tenure≥0 "
          f"({df['pledge_id'].nunique():,} pledges validados).")


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error (ECE). 0 = calibración perfecta."""
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(probs)
    err = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        err += mask.sum() / total * abs(probs[mask].mean() - labels[mask].mean())
    return err


def main():
    CUTOFF       = get_cutoff_date()
    TEST_YEAR    = CUTOFF.year          # año de test = año del corte
    TRAIN_END    = TEST_YEAR - 1        # último año de train
    CUTOFF_MONTH = CUTOFF.replace(day=1)  # snapshot del mes de corte

    print("[1/4] Cargando dataset pledge × mes...")
    df = pd.read_parquet(DATASET)
    print(f"  Filas: {len(df):,}")
    print(f"  Pledges únicos: {df['pledge_id'].nunique():,}")
    print(f"  Tasa churn 30d: {df['label_churn_30d'].mean():.4f}")

    # === Coherencia dataset ↔ fecha de corte ===
    # El dataset de Motor 2 depende del corte: 01_motor2_dataset.py genera 24
    # snapshots mensuales terminando en el mes de corte. Si el parquet fue
    # generado con otra fecha, entrenar sería un error silencioso (run_all.sh
    # SALTA las etapas cuya salida ya existe). Se aborta con instrucciones.
    snap_min, snap_max = df["snapshot_ts"].min(), df["snapshot_ts"].max()
    print(f"  Snapshots del dataset: {snap_min:%Y-%m} → {snap_max:%Y-%m}")
    if snap_max != CUTOFF_MONTH:
        raise SystemExit(
            f"\n[ERROR] El dataset termina en el snapshot {snap_max:%Y-%m-%d} pero el "
            f"corte pedido es {CUTOFF.date()} (mes de corte {CUTOFF_MONTH:%Y-%m-%d}).\n"
            f"        Regenera las features con la MISMA fecha:\n"
            f"          python3 scripts/02_features_motor2/01_motor2_dataset.py "
            f"--cutoff-date {CUTOFF.date()}\n"
            f"        (o corre el pipeline completo con "
            f"FORCE=1 CUTOFF_DATE={CUTOFF.date()} bash scripts/run_all.sh)\n")

    # === Tratamiento de nulos por feature ===
    for c in ["last_payment_method", "last_bank_name", "last_gateway"]:
        df[c] = df[c].astype(str).fillna("Unknown")
    df["last_payment_amount"] = df["last_payment_amount"].fillna(0).astype("float32")
    df["days_since_last_attempt"] = df["days_since_last_attempt"].fillna(999).astype("float32")
    df["last_attempt_success"] = df["last_attempt_success"].fillna(-1).astype("int8")

    # === Re-orden canónico ===
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    # === Split temporal estricto (derivado del corte) ===
    train = df[df["snapshot_ts"].dt.year <= TRAIN_END].copy()
    test  = df[df["snapshot_ts"].dt.year == TEST_YEAR].copy()
    NOMBRE_TRAIN = f"train (snapshots <= {TRAIN_END})"
    NOMBRE_TEST  = f"test (snapshots {TEST_YEAR})"
    print(f"  {NOMBRE_TRAIN}: {len(train):,}")
    print(f"  {NOMBRE_TEST}: {len(test):,}")

    if len(train) == 0:
        raise SystemExit(
            f"\n[ERROR] Train vacío: no hay snapshots de años <= {TRAIN_END} en el dataset "
            f"(rango disponible: {snap_min:%Y-%m} → {snap_max:%Y-%m}).\n"
            f"        Con corte {CUTOFF.date()} la ventana de 24 meses no alcanza a cubrir "
            f"un año completo de train.\n")
    if train["label_churn_30d"].sum() == 0:
        raise SystemExit(
            f"\n[ERROR] Train sin casos positivos de churn — no se puede entrenar.\n")
    if len(test) == 0:
        raise SystemExit(
            f"\n[ERROR] Test vacío: no hay snapshots del año {TEST_YEAR} en el dataset "
            f"(rango disponible: {snap_min:%Y-%m} → {snap_max:%Y-%m}).\n")

    print(f"  Meses train: {train['snapshot_ts'].min():%Y-%m} → {train['snapshot_ts'].max():%Y-%m}")
    print(f"  Meses test:  {test['snapshot_ts'].min():%Y-%m} → {test['snapshot_ts'].max():%Y-%m}")
    print(f"  Tasa churn train: {train['label_churn_30d'].mean():.4f}")
    print(f"  Tasa churn test:  {test['label_churn_30d'].mean():.4f}")

    # === Censura del label (ventana de 30 días aún abierta) ===
    # El label de un snapshot es OBSERVABLE solo si su ventana de 30 días cerró
    # antes del corte (snapshot + 30d <= CUTOFF). Los snapshots del mes de corte
    # tienen la ventana abierta: como churn solo aparecen las cancelaciones YA
    # registradas en el CRM al momento del corte, así que la tasa observada
    # subestima la real y cualquier métrica sobre esas filas es optimista.
    label_observable = (test["snapshot_ts"] + pd.Timedelta(days=30)) <= CUTOFF
    n_censurado = int((~label_observable).sum())
    if n_censurado:
        print(f"  [AVISO] {n_censurado:,} filas de test "
              f"({100 * n_censurado / len(test):.1f}%) tienen la ventana de 30d abierta "
              f"al {CUTOFF.date()} → label parcialmente observado.")

    # === ¿Se puede evaluar el test? ===
    # roc_auc_score exige ambas clases presentes. Con un corte en el mes en
    # curso el test puede ser un único snapshot con 0 churns registrados: en ese
    # caso se pasa a MODO SCORING (entrena, calibra, escribe predicciones, sin
    # métricas de test).
    PUEDE_EVALUAR = int(test["label_churn_30d"].nunique()) > 1
    MODO = "evaluacion" if PUEDE_EVALUAR else "scoring"
    if not PUEDE_EVALUAR:
        print(f"  [AVISO] El test ({TEST_YEAR}) tiene una sola clase "
              f"(churns={int(test['label_churn_30d'].sum())}) → MODO SCORING: "
              f"se entrena y se escriben predicciones, sin métricas de test.")

    # === Validación de integridad temporal (anti-leakage) ===
    validate_temporal_integrity(train, NOMBRE_TRAIN)
    validate_temporal_integrity(test, NOMBRE_TEST)

    # === Diagnóstico de overlap pledges train↔test (esperado, NO leakage) ===
    train_pledges = set(train["pledge_id"].unique())
    test_pledges  = set(test["pledge_id"].unique())
    overlap = len(train_pledges & test_pledges)
    print(f"  Pledges en train: {len(train_pledges):,}")
    print(f"  Pledges en test:  {len(test_pledges):,}")
    print(f"  Overlap (esperado, pledges activos en ambos años): {overlap:,} "
          f"({100*overlap/len(test_pledges):.1f}% del test)")

    feat_num = ["tenure_days", "n_intentos_30d", "n_fallos_30d", "streak_actual",
                "success_rate_hist", "cum_intentos", "cum_exitos", "cum_fallos",
                "days_since_last_attempt", "last_payment_amount", "last_attempt_success"]
    feat_cat = ["last_payment_method", "last_bank_name", "last_gateway"]
    features = feat_num + feat_cat

    print("[2/4] Baselines triviales...")
    baselines = {}

    if PUEDE_EVALUAR:
        # B0: predicción CONSTANTE = tasa global de churn en train.
        p0 = np.full(len(test), train["label_churn_30d"].mean(), dtype=float)
        baselines["B0_constante"] = {
            "auc":   float(roc_auc_score(test["label_churn_30d"], p0)) if test["label_churn_30d"].sum() > 0 else 0.5,
            "brier": float(brier_score_loss(test["label_churn_30d"], p0)),
        }

        # B1: lookup por streak_actual (capped a 15).
        train["streak_cap"] = train["streak_actual"].clip(upper=15)
        lookup_streak = train.groupby("streak_cap")["label_churn_30d"].mean().to_dict()
        p1 = test["streak_actual"].clip(upper=15).map(lookup_streak).fillna(train["label_churn_30d"].mean()).values.astype(float)
        baselines["B1_lookup_streak"] = {
            "auc":   float(roc_auc_score(test["label_churn_30d"], p1)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p1)),
        }

        # B2: lookup por tenure (antigüedad del pledge en bins).
        bins = [-1, 30, 90, 180, 365, 730, 10000]
        train["tenure_bin"] = pd.cut(train["tenure_days"], bins=bins).astype(str)
        test["tenure_bin"]  = pd.cut(test["tenure_days"],  bins=bins).astype(str)
        lookup_tenure = train.groupby("tenure_bin")["label_churn_30d"].mean().to_dict()
        p2 = test["tenure_bin"].map(lookup_tenure).fillna(train["label_churn_30d"].mean()).values.astype(float)
        baselines["B2_lookup_tenure"] = {
            "auc":   float(roc_auc_score(test["label_churn_30d"], p2)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p2)),
        }
        print(f"  B0: AUC={baselines['B0_constante']['auc']:.4f}  Brier={baselines['B0_constante']['brier']:.4f}")
        print(f"  B1 (streak): AUC={baselines['B1_lookup_streak']['auc']:.4f}  Brier={baselines['B1_lookup_streak']['brier']:.4f}")
        print(f"  B2 (tenure): AUC={baselines['B2_lookup_tenure']['auc']:.4f}  Brier={baselines['B2_lookup_tenure']['brier']:.4f}")
    else:
        print("  (omitidos: el test no tiene ambas clases — modo scoring)")

    print("[3/4] Entrenando CatBoost (v3: sin pesos de clase, lr=0.03)...")
    # Validación interna para early stopping Y para ajustar el calibrador.
    # CAMBIO v9: partición TEMPORAL (últimos 2 meses del train) en vez del
    # último 10% por posición. El corte posicional ordenaba por pledge_id, que
    # al ser cronológico dejaba un val con 5,90% de churn contra 3,48% del test:
    # cortaba el entrenamiento en 59 árboles y sesgaba el calibrador hacia
    # arriba. Ver docs/07_Informe_calibracion.md.
    tr_idx, val_idx = split_validacion_temporal(train)

    # CatBoost v3 — sin auto_class_weights, learning_rate más bajo:
    #   iterations=800        ← tope alto; early stopping decide el corte real
    #   learning_rate=0.03    ← convergencia más lenta y robusta
    #   loss_function=Logloss ← clasificación binaria
    #   eval_metric=AUC       ← se monitorea AUC en validación interna
    #   od_type=Iter, od_wait=15 → early stopping
    #   SIN auto_class_weights → el calibrador lleva las probabilidades a la
    #     tasa real sin distorsionar el ranking (ver experimento 02b).
    #   boost_from_average=True → arranca en la tasa base en vez de 0,5. Es el
    #     default de CatBoost para RMSE pero NO para Logloss. Sin esto el modelo
    #     arranca prediciendo "moneda al aire" y el early stopping lo corta
    #     antes de que las probabilidades crudas bajen a su valor real.
    model = CatBoostClassifier(
        iterations=800,
        boost_from_average=True,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=feat_cat,
        random_seed=42,
        verbose=100,
        thread_count=-1,
        od_type="Iter",
        od_wait=15,
    )
    model.fit(
        train.loc[tr_idx, features], train.loc[tr_idx, "label_churn_30d"],
        eval_set=(train.loc[val_idx, features], train.loc[val_idx, "label_churn_30d"]),
        use_best_model=True,
    )
    model.save_model(str(OUT / "motor2_catboost.cbm"))
    print(f"  Árboles usados (best_iteration): {model.get_best_iteration()}")

    # === Calibración isotónica ===
    # Sin Balanced el sesgo crudo es pequeño, pero mantenemos el calibrador:
    # es barato, entrega la columna prob_churn lista para producción y protege
    # ante drift. Se entrena sobre el mismo val set del early stopping.
    print("  [calibración] Entrenando IsotonicRegression sobre validation set...")
    prob_val_cruda = model.predict_proba(train.loc[val_idx, features])[:, 1]
    y_val = train.loc[val_idx, "label_churn_30d"].values

    calibrador = IsotonicRegression(out_of_bounds="clip")
    calibrador.fit(prob_val_cruda, y_val)

    with open(OUT / "motor2_calibrador.pkl", "wb") as f:
        pickle.dump(calibrador, f)

    prob_val_cal = calibrador.transform(prob_val_cruda)
    ece_val_antes   = ece(prob_val_cruda, y_val)
    ece_val_despues = ece(prob_val_cal,   y_val)
    print(f"  [calibración] ECE val — antes: {ece_val_antes:.4f}  →  después: {ece_val_despues:.4f}  "
          f"(mejora: {ece_val_antes - ece_val_despues:+.4f})")

    print("[4/4] Evaluando..." if PUEDE_EVALUAR else "[4/4] Puntuando (modo scoring)...")
    y_prob_cruda = model.predict_proba(test[features])[:, 1]
    y_prob       = calibrador.transform(y_prob_cruda)   # probabilidades calibradas

    metricas_test = None
    if PUEDE_EVALUAR:
        auc    = roc_auc_score(test["label_churn_30d"], y_prob)
        pr_auc = average_precision_score(test["label_churn_30d"], y_prob)
        brier         = brier_score_loss(test["label_churn_30d"], y_prob)
        brier_crudo   = brier_score_loss(test["label_churn_30d"], y_prob_cruda)

        ece_test_antes   = ece(y_prob_cruda, test["label_churn_30d"].values)
        ece_test_despues = ece(y_prob,       test["label_churn_30d"].values)
        print(f"  [calibración] ECE test — antes: {ece_test_antes:.4f}  →  después: {ece_test_despues:.4f}")

        # === Métrica operacional: Recall y Precision @ top 20% ===
        threshold_top20 = np.quantile(y_prob, 0.80)
        pred_top20      = (y_prob >= threshold_top20).astype(int)
        recall_top20    = (pred_top20 & test["label_churn_30d"].values).sum() / max(test["label_churn_30d"].sum(), 1)
        precision_top20 = (pred_top20 & test["label_churn_30d"].values).sum() / max(pred_top20.sum(), 1)

        confiabilidad = tabla_confiabilidad(y_prob, test["label_churn_30d"].values)
        imprimir_confiabilidad(confiabilidad, test["label_churn_30d"].mean())

        metricas_test = {
            "auc":              float(auc),
            "pr_auc":           float(pr_auc),
            "brier_calibrado":  float(brier),
            "brier_crudo":      float(brier_crudo),
            "ece_test_antes_calibrar":   float(ece_test_antes),
            "ece_test_despues_calibrar": float(ece_test_despues),
            "recall_top20":     float(recall_top20),
            "precision_top20":  float(precision_top20),
            # True → parte (o todo) el test tiene la ventana de 30d abierta al
            # corte: las métricas de arriba son optimistas, no comparables con
            # las de un corte a fin de año.
            "label_censurado": bool(n_censurado > 0),
            # Lo que negocio necesita saber: en el top N%, ¿lo prometido
            # coincide con lo ocurrido? Ver docs/07_Informe_calibracion.md.
            "confiabilidad_tramo_alto": confiabilidad,
        }
    else:
        print(f"  Sin métricas de test: {len(test):,} filas con una sola clase "
              f"(el snapshot {test['snapshot_ts'].max():%Y-%m} todavía no tiene desenlace). "
              f"Usar las métricas de validación interna como referencia.")

    importance = pd.DataFrame({
        "feature":    features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    results = {
        "n_train":          len(train),
        "n_test":           len(test),
        "n_train_used":     len(train),
        "tasa_churn_train": float(train["label_churn_30d"].mean()),
        "tasa_churn_test":  float(test["label_churn_30d"].mean()),
        "pledges_overlap_train_test": overlap,
        "pledges_overlap_pct_test":   100 * overlap / len(test_pledges),
        # Trazabilidad del corte: con qué fecha se entrenó este modelo y qué
        # ventana temporal cubren train y test.
        "corte": {
            "cutoff_date":      str(CUTOFF.date()),
            "test_year":        TEST_YEAR,
            "train_years":      f"<= {TRAIN_END}",
            "snapshots_dataset": [f"{snap_min:%Y-%m}", f"{snap_max:%Y-%m}"],
            "snapshots_train":  [f"{train['snapshot_ts'].min():%Y-%m}", f"{train['snapshot_ts'].max():%Y-%m}"],
            "snapshots_test":   [f"{test['snapshot_ts'].min():%Y-%m}", f"{test['snapshot_ts'].max():%Y-%m}"],
            "modo":             MODO,
            "n_test_label_censurado":   n_censurado,
            "pct_test_label_censurado": 100 * n_censurado / len(test),
        },
        "config_v3": {
            "sample_frac":           1.0,
            "auto_class_weights":    None,
            "learning_rate":         0.03,
            "iterations":            800,
            "best_iteration":        int(model.get_best_iteration()),
            "orden":                 "pledge_id, snapshot_ts",
            "boost_from_average":    True,
            "validacion":            "temporal (ultimos 2 meses del train)",
            "anti_leakage_validado": True,
            "calibracion":           "IsotonicRegression",
        },
        # None en modo scoring (test sin desenlace observable todavía).
        "catboost": metricas_test,
        "calibracion": {
            "metodo":          "IsotonicRegression",
            "entrenado_en":    "validation set (ultimo 10% del train)",
            "archivo":         "motor2_calibrador.pkl",
            "ece_val_antes":   float(ece_val_antes),
            "ece_val_despues": float(ece_val_despues),
            "mejora_ece_val":  float(ece_val_antes - ece_val_despues),
        },
        "baselines": baselines,
        "lift_auc_vs_b1": float(auc - baselines["B1_lookup_streak"]["auc"]) if PUEDE_EVALUAR else None,
        "lift_auc_vs_b2": float(auc - baselines["B2_lookup_tenure"]["auc"]) if PUEDE_EVALUAR else None,
        "feature_importance": importance.to_dict(orient="records"),
    }

    with open(OUT / "resultados_motor2.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    importance.to_csv(OUT / "feature_importance_motor2.csv", index=False)

    # Guardar predicciones de test (para Motor 3 / tabla maestra)
    # Se escriben en dos formatos:
    #   .parquet → insumo del pipeline (Motor 3), compacto y tipado.
    #   .csv     → mismo contenido, legible por el equipo (Excel/hojas de cálculo).
    test_out = test[["pledge_id", "snapshot_ts", "label_churn_30d"]].copy()
    test_out["prob_churn"]       = y_prob.astype("float32")        # calibrada ← usar esta
    test_out["prob_churn_cruda"] = y_prob_cruda.astype("float32")  # cruda     ← solo referencia
    test_out = test_out.sort_values(["prob_churn"], ascending=False).reset_index(drop=True)
    test_out.to_parquet(OUT / "test_predictions_motor2.parquet", index=False)
    test_out.to_csv(OUT / "test_predictions_motor2.csv", index=False)

    print(f"\n  Corte: {CUTOFF.date()}  |  train <= {TRAIN_END}  |  test {TEST_YEAR}  |  modo: {MODO}")
    if PUEDE_EVALUAR:
        print(f"  CatBoost AUC (calibrado):    {auc:.4f}")
        print(f"  Baseline B1 AUC (streak):    {baselines['B1_lookup_streak']['auc']:.4f}")
        print(f"  Baseline B2 AUC (tenure):    {baselines['B2_lookup_tenure']['auc']:.4f}")
        print(f"  Lift vs B1:                  {auc - baselines['B1_lookup_streak']['auc']:+.4f}")
        print(f"  Lift vs B2:                  {auc - baselines['B2_lookup_tenure']['auc']:+.4f}")
        print(f"  PR-AUC:                      {pr_auc:.4f}")
        print(f"  Brier crudo: {brier_crudo:.4f}  →  calibrado: {brier:.4f}")
        print(f"  ECE test   — antes: {ece_test_antes:.4f}  →  después: {ece_test_despues:.4f}")
        print(f"  Recall @ top 20%:            {recall_top20:.4f}")
        print(f"  Precision @ top 20%:         {precision_top20:.4f}")
        if n_censurado:
            print(f"  [AVISO] {100 * n_censurado / len(test):.1f}% del test con ventana de 30d "
                  f"abierta al corte → métricas optimistas, no comparables con un corte de fin de año.")
    else:
        print(f"  Métricas de test: N/A (modo scoring — el test aún no tiene desenlace)")
        print(f"  Referencia de calidad: ECE val {ece_val_antes:.4f} → {ece_val_despues:.4f}")
    print(f"  Predicciones escritas: {len(test_out):,} filas "
          f"({test['snapshot_ts'].min():%Y-%m} → {test['snapshot_ts'].max():%Y-%m})")
    print(f"  Top features: {importance.head(5).to_dict('records')}")


if __name__ == "__main__":
    main()
