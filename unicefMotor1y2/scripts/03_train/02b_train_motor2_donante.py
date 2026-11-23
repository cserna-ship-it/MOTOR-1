#!/usr/bin/env python3
"""
02b_train_motor2_donante.py — Entrenamiento Motor 2 a nivel DONANTE

QUÉ CAMBIA RESPECTO A 02_train_motor2.py (pledge-level)
--------------------------------------------------------
Mismo algoritmo (CatBoost), misma filosofía (sin auto_class_weights, lr=0.03,
calibración isotónica, split temporal out-of-time) — lo único que cambia es la
UNIDAD de observación: antes era (pledge, snapshot), ahora es (contact_id,
snapshot). Todo lo demás del pipeline se mantiene igual a propósito: ya sabemos
que esa parte funciona.

Se conservan también los guardarraíles del pledge-level, adaptados:
  - coherencia dataset ↔ fecha de corte (aborta si el parquet es de otro corte)
  - censura del label cuando la ventana de 30d aún no cerró
  - MODO SCORING cuando el test tiene una sola clase
  - resultados_motor2_donante.json con la misma estructura

LAS MÉTRICAS NO SON COMPARABLES CON LAS DEL MODELO PLEDGE-LEVEL
----------------------------------------------------------------
El label de donante es MÁS RARO que el de pledge (hay que cancelar todos los
pledges, no uno). Con una prevalencia menor, la PR-AUC baja mecánicamente
aunque el modelo sea igual de bueno. Comparar 0.367 (pledge) contra la PR-AUC
de donante sería un error de lectura: son dos problemas distintos.

Lee:    data/features/motor2_donante_month.parquet   (output de 01b)
Genera: results/motor2_donante/                      (carpeta NUEVA, no pisa
                                                      results/motor2/)
        results/motor2_donante/motor2_donante_cutoff.json

PUNTO DE CORTE (CRUCE SENSIBILIDAD = ESPECIFICIDAD)
---------------------------------------------------
Al final del entrenamiento se calcula el umbral donde se cruzan las curvas de
sensibilidad y especificidad (argmin |Sens − Espec|) sobre el TEST (out-of-time)
y se guarda como artefacto, amarrado al .cbm por su sha256. 04b lo lee y agrega la
columna `alerta` (0/1). Si el test tiene una sola clase (modo scoring), el
umbral se calcula sobre el validation set temporal — es lo único con desenlace
conocido que queda.

Para generar SOLO el cutoff con el modelo ya entrenado (sin reentrenar), usando
las predicciones de test que este mismo script dejó escritas:

    python3 scripts/03_train/02b_train_motor2_donante.py --solo-cutoff
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/
from config import (get_cutoff_date, imprimir_confiabilidad,
                    split_validacion_temporal, tabla_confiabilidad)
from motor2_donante_common import (
    CAT_FEATURES_DONANTE, FEATURES_DONANTE, aplicar_cutoff,
    METODO_CUTOFF, calcular_cutoff, guardar_cutoff, preparar_features_donante,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/features/motor2_donante_month.parquet"
OUT = ROOT / "results/motor2_donante"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "motor2_donante_catboost.cbm"
TEST_PRED_PATH = OUT / "test_predictions_motor2_donante.parquet"


def imprimir_cutoff(info, path):
    m = info["matriz_confusion"]
    print(f"  [cutoff {info['metodo']}] umbral = {info['cutoff']:.4f}  "
          f"(calculado en: {info['calculado_en']})")
    print(f"    sensibilidad {info['sensibilidad']:.1%}  |  especificidad "
          f"{info['especificidad']:.1%}  |  J = {info['youden_j']:.4f}")
    print(f"    en alerta: {info['pct_donantes_en_alerta']:.1%} de los donantes  |  "
          f"precisión {info['precision']:.1%}  (lift {info['lift']:.1f}x)")
    print(f"    VP={m['VP']:,}  FP={m['FP']:,}  FN={m['FN']:,}  VN={m['VN']:,}")
    print(f"  [OK] {path}")


def solo_cutoff():
    """Recalcula el cutoff con el modelo actual, sin reentrenar."""
    for p in (MODEL_PATH, TEST_PRED_PATH):
        if not p.exists():
            raise SystemExit(f"\n[ERROR] Falta {p}. Entrena primero (sin --solo-cutoff).\n")
    print(f"[--solo-cutoff] Leyendo {TEST_PRED_PATH.name} (predicciones del modelo actual)...")
    t = pd.read_parquet(TEST_PRED_PATH, columns=["label_churn_30d", "prob_churn_donante"])
    if t["label_churn_30d"].nunique() < 2:
        raise SystemExit("\n[ERROR] Las predicciones de test tienen una sola clase: "
                         "reentrena para calcular el cutoff sobre el validation set.\n")
    info = calcular_cutoff(t["label_churn_30d"].values, t["prob_churn_donante"].values)
    # La huella amarra el cutoff al .cbm actual: solo es válido si las
    # predicciones de test las generó ESE modelo (las escribe esta misma
    # corrida de 02b, así que van juntas mientras nadie las mezcle a mano).
    path = guardar_cutoff(info, OUT, MODEL_PATH, calculado_en="test (predicciones existentes)")
    imprimir_cutoff({**info, "calculado_en": "test (predicciones existentes)"}, path)

    # Deja el cutoff también en el JSON de resultados, igual que una corrida completa.
    res_path = OUT / "resultados_motor2_donante.json"
    if res_path.exists():
        with open(res_path) as f:
            results = json.load(f)
        results.pop("cutoff_youden", None)   # nombre anterior de la llave
        # Texto heredado: el calibrador siempre se entrenó sobre la validación temporal.
        results.setdefault("calibracion", {})["entrenado_en"] = (
            "validation set temporal (ultimos 2 meses del train)")
        results["punto_corte"] = {**info, "archivo": path.name,
                                    "calculado_en": "test (predicciones existentes)"}
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  [OK] cutoff agregado a {res_path.name}")

# La llave de orden canónico pasa de pledge_id a contact_id: es la unidad que
# ahora define "una serie temporal" en el dataset.
SORT_KEYS = ["contact_id", "snapshot_ts"]


def validate_temporal_integrity_donante(df, nombre):
    """
    Anti-leakage a nivel donante — ADAPTADA, no copiada tal cual del
    pledge-level.

    Se mantienen como assert las dos invariantes que SIGUEN siendo verdad sin
    importar cuántos pledges tenga el donante:
      1. days_since_last_attempt_min nunca es negativo.
      2. tenure_max / tenure_min nunca son negativos.

    Se BAJA a diagnóstico impreso la invariante de "los acumulados nunca
    decrecen". A nivel donante SÍ puede pasar legítimamente: si un donante tenía
    2 pledges y uno cancela, ese pledge sale del conjunto activo en los
    snapshots siguientes y la suma cum_intentos_total puede bajar sin que eso
    sea fuga de información — es un cambio real en quién compone el grupo.
    La garantía de fondo (que CADA pledge solo ve su propio pasado) ya se validó
    aguas arriba, en el pipeline pledge-level.
    """
    mask_real = df["days_since_last_attempt_min"] < 999
    neg = int((df.loc[mask_real, "days_since_last_attempt_min"] < 0).sum())
    assert neg == 0, (
        f"[{nombre}] {neg} filas con days_since_last_attempt_min negativo → "
        "revisar merge_asof aguas arriba (pledge-level).")

    for col in ["tenure_max", "tenure_min"]:
        neg_t = int((df[col] < 0).sum())
        assert neg_t == 0, (
            f"[{nombre}] {neg_t} filas con {col} negativo → snapshot anterior "
            "al inicio efectivo de algún pledge del donante.")

    df_sorted = df.sort_values(SORT_KEYS)
    dec = int((df_sorted.groupby("contact_id", sort=False)["cum_intentos_total"]
               .diff() < 0).sum())
    pct = 100 * dec / len(df_sorted) if len(df_sorted) else 0
    print(f"  [OK anti-leakage] {nombre}: days≥0, tenure≥0 "
          f"({df['contact_id'].nunique():,} donantes validados).")
    print(f"  [diagnóstico] cum_intentos_total decrece en {dec:,} filas ({pct:.2f}%) "
          f"— esperado cuando un pledge sale del conjunto activo del donante, "
          f"NO es un fallo de anti-leakage por sí solo.")


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
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solo-cutoff", action="store_true",
                        help="Solo recalcula motor2_donante_cutoff.json con el modelo actual.")
    args, _ = parser.parse_known_args()   # --cutoff-date lo lee config.py
    if args.solo_cutoff:
        solo_cutoff()
        return

    CUTOFF = get_cutoff_date()
    TEST_YEAR = CUTOFF.year
    TRAIN_END = TEST_YEAR - 1
    CUTOFF_MONTH = CUTOFF.replace(day=1)

    if not DATASET.exists():
        raise SystemExit(
            f"\n[ERROR] No existe {DATASET}\n"
            f"        Genera primero el dataset de donante:\n"
            f"          python3 scripts/02_features_motor2/01b_motor2_dataset_donante.py\n")

    print("[1/4] Cargando dataset donante × mes...")
    df = pd.read_parquet(DATASET)
    print(f"  Filas: {len(df):,}  |  donantes únicos: {df['contact_id'].nunique():,}")
    print(f"  Tasa churn 30d (donante): {df['label_churn_30d'].mean():.4f}")

    # === Coherencia dataset ↔ fecha de corte ===
    # Mismo guardarraíl que el pledge-level: si el parquet fue generado con otro
    # corte, entrenar sería un error silencioso (run_all.sh salta las etapas
    # cuya salida ya existe).
    snap_min, snap_max = df["snapshot_ts"].min(), df["snapshot_ts"].max()
    print(f"  Snapshots del dataset: {snap_min:%Y-%m} → {snap_max:%Y-%m}")
    if snap_max != CUTOFF_MONTH:
        raise SystemExit(
            f"\n[ERROR] El dataset termina en el snapshot {snap_max:%Y-%m-%d} pero el "
            f"corte pedido es {CUTOFF.date()} (mes de corte {CUTOFF_MONTH:%Y-%m-%d}).\n"
            f"        Regenera las features con la MISMA fecha:\n"
            f"          python3 scripts/02_features_motor2/01_motor2_dataset.py "
            f"--cutoff-date {CUTOFF.date()}\n"
            f"          python3 scripts/02_features_motor2/01b_motor2_dataset_donante.py\n")

    # Imputación y tipos — la MISMA función que usa 04_score_motor2_donante.py.
    df = preparar_features_donante(df)
    df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    train = df[df["snapshot_ts"].dt.year <= TRAIN_END].copy()
    test = df[df["snapshot_ts"].dt.year == TEST_YEAR].copy()
    NOMBRE_TRAIN = f"train (snapshots <= {TRAIN_END})"
    NOMBRE_TEST = f"test (snapshots {TEST_YEAR})"
    print(f"  {NOMBRE_TRAIN}: {len(train):,}")
    print(f"  {NOMBRE_TEST}: {len(test):,}")

    if len(train) == 0:
        raise SystemExit(
            f"\n[ERROR] Train vacío: no hay snapshots de años <= {TRAIN_END} "
            f"(rango disponible: {snap_min:%Y-%m} → {snap_max:%Y-%m}).\n")
    if train["label_churn_30d"].sum() == 0:
        raise SystemExit("\n[ERROR] Train sin casos positivos de churn — no se puede entrenar.\n")
    if len(test) == 0:
        raise SystemExit(
            f"\n[ERROR] Test vacío: no hay snapshots del año {TEST_YEAR} "
            f"(rango disponible: {snap_min:%Y-%m} → {snap_max:%Y-%m}).\n")

    print(f"  Meses train: {train['snapshot_ts'].min():%Y-%m} → {train['snapshot_ts'].max():%Y-%m}")
    print(f"  Meses test:  {test['snapshot_ts'].min():%Y-%m} → {test['snapshot_ts'].max():%Y-%m}")
    print(f"  Tasa churn train: {train['label_churn_30d'].mean():.4f}")
    print(f"  Tasa churn test:  {test['label_churn_30d'].mean():.4f}")

    # === Censura del label (ventana de 30 días aún abierta) ===
    label_observable = (test["snapshot_ts"] + pd.Timedelta(days=30)) <= CUTOFF
    n_censurado = int((~label_observable).sum())
    if n_censurado:
        print(f"  [AVISO] {n_censurado:,} filas de test "
              f"({100 * n_censurado / len(test):.1f}%) tienen la ventana de 30d abierta "
              f"al {CUTOFF.date()} → label parcialmente observado.")

    PUEDE_EVALUAR = int(test["label_churn_30d"].nunique()) > 1
    MODO = "evaluacion" if PUEDE_EVALUAR else "scoring"
    if not PUEDE_EVALUAR:
        print(f"  [AVISO] El test ({TEST_YEAR}) tiene una sola clase "
              f"(churns={int(test['label_churn_30d'].sum())}) → MODO SCORING: "
              f"se entrena y se escriben predicciones, sin métricas de test.")

    validate_temporal_integrity_donante(train, NOMBRE_TRAIN)
    validate_temporal_integrity_donante(test, NOMBRE_TEST)

    # Overlap medido en DONANTES, no en pledges.
    train_donantes = set(train["contact_id"].unique())
    test_donantes = set(test["contact_id"].unique())
    overlap = len(train_donantes & test_donantes)
    print(f"  Donantes en train: {len(train_donantes):,}")
    print(f"  Donantes en test:  {len(test_donantes):,}")
    print(f"  Overlap (esperado, donantes activos en ambos años): {overlap:,} "
          f"({100 * overlap / len(test_donantes):.1f}% del test)")

    features = FEATURES_DONANTE
    feat_cat = CAT_FEATURES_DONANTE

    print("[2/4] Baselines triviales (adaptados a donante)...")
    baselines = {}
    if PUEDE_EVALUAR:
        p0 = np.full(len(test), train["label_churn_30d"].mean(), dtype=float)
        baselines["B0_constante"] = {
            "auc": float(roc_auc_score(test["label_churn_30d"], p0)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p0)),
        }

        # B1: lookup por la peor racha de fallos entre los pledges del donante.
        train["streak_cap"] = train["streak_max"].clip(upper=15)
        lookup_streak = train.groupby("streak_cap")["label_churn_30d"].mean().to_dict()
        p1 = (test["streak_max"].clip(upper=15).map(lookup_streak)
              .fillna(train["label_churn_30d"].mean()).values.astype(float))
        baselines["B1_lookup_streak_max"] = {
            "auc": float(roc_auc_score(test["label_churn_30d"], p1)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p1)),
        }

        # B2: lookup por antigüedad del pledge más viejo del donante.
        bins = [-1, 30, 90, 180, 365, 730, 10000]
        train["tenure_bin"] = pd.cut(train["tenure_max"], bins=bins).astype(str)
        test["tenure_bin"] = pd.cut(test["tenure_max"], bins=bins).astype(str)
        lookup_tenure = train.groupby("tenure_bin")["label_churn_30d"].mean().to_dict()
        p2 = (test["tenure_bin"].map(lookup_tenure)
              .fillna(train["label_churn_30d"].mean()).values.astype(float))
        baselines["B2_lookup_tenure_max"] = {
            "auc": float(roc_auc_score(test["label_churn_30d"], p2)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p2)),
        }

        # B3: n_pledges_activos — la señal NUEVA del cambio a donante. Si el
        # modelo no le gana claramente a esto, la agregación no está aportando.
        lookup_np = train.groupby(train["n_pledges_activos"].clip(upper=5))[
            "label_churn_30d"].mean().to_dict()
        p3 = (test["n_pledges_activos"].clip(upper=5).map(lookup_np)
              .fillna(train["label_churn_30d"].mean()).values.astype(float))
        baselines["B3_lookup_n_pledges"] = {
            "auc": float(roc_auc_score(test["label_churn_30d"], p3)),
            "brier": float(brier_score_loss(test["label_churn_30d"], p3)),
        }
        for k, v in baselines.items():
            print(f"  {k}: AUC={v['auc']:.4f}  Brier={v['brier']:.4f}")
    else:
        print("  (omitidos: el test no tiene ambas clases — modo scoring)")

    print("[3/4] Entrenando CatBoost (misma config de producción: sin pesos, lr=0.03)...")
    # Partición TEMPORAL (últimos 2 meses del train), no el último 10% por
    # posición: ordenar por contact_id equivale a ordenar por antigüedad del
    # donante, y eso dejaba un val con 5,67% de churn contra 3,29% del test.
    # Ver docs/07_Informe_calibracion.md.
    tr_idx, val_idx = split_validacion_temporal(train)

    model = CatBoostClassifier(
        iterations=800,
        # Arranca en la tasa base, no en 0,5. Ver la nota en 02_train_motor2.py.
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
    model.save_model(str(MODEL_PATH))
    print(f"  Árboles usados (best_iteration): {model.get_best_iteration()}")

    print("  [calibración] Entrenando IsotonicRegression sobre validation set...")
    prob_val_cruda = model.predict_proba(train.loc[val_idx, features])[:, 1]
    y_val = train.loc[val_idx, "label_churn_30d"].values
    calibrador = IsotonicRegression(out_of_bounds="clip").fit(prob_val_cruda, y_val)
    with open(OUT / "motor2_donante_calibrador.pkl", "wb") as f:
        pickle.dump(calibrador, f)

    prob_val_cal = calibrador.transform(prob_val_cruda)
    ece_val_antes = ece(prob_val_cruda, y_val)
    ece_val_despues = ece(prob_val_cal, y_val)
    print(f"  [calibración] ECE val — antes: {ece_val_antes:.4f}  →  después: "
          f"{ece_val_despues:.4f}  (mejora: {ece_val_antes - ece_val_despues:+.4f})")

    print("[4/4] Evaluando..." if PUEDE_EVALUAR else "[4/4] Puntuando (modo scoring)...")
    y_prob_cruda = model.predict_proba(test[features])[:, 1]
    y_prob = calibrador.transform(y_prob_cruda)

    metricas_test = None
    if PUEDE_EVALUAR:
        auc = roc_auc_score(test["label_churn_30d"], y_prob)
        pr_auc = average_precision_score(test["label_churn_30d"], y_prob)
        brier = brier_score_loss(test["label_churn_30d"], y_prob)
        brier_crudo = brier_score_loss(test["label_churn_30d"], y_prob_cruda)
        ece_test_antes = ece(y_prob_cruda, test["label_churn_30d"].values)
        ece_test_despues = ece(y_prob, test["label_churn_30d"].values)
        print(f"  [calibración] ECE test — antes: {ece_test_antes:.4f}  →  "
              f"después: {ece_test_despues:.4f}")

        threshold_top20 = np.quantile(y_prob, 0.80)
        pred_top20 = (y_prob >= threshold_top20).astype(int)
        recall_top20 = ((pred_top20 & test["label_churn_30d"].values).sum()
                        / max(test["label_churn_30d"].sum(), 1))
        precision_top20 = ((pred_top20 & test["label_churn_30d"].values).sum()
                           / max(pred_top20.sum(), 1))

        confiabilidad = tabla_confiabilidad(y_prob, test["label_churn_30d"].values)
        imprimir_confiabilidad(confiabilidad, test["label_churn_30d"].mean())

        metricas_test = {
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "brier_calibrado": float(brier),
            "brier_crudo": float(brier_crudo),
            "ece_test_antes_calibrar": float(ece_test_antes),
            "ece_test_despues_calibrar": float(ece_test_despues),
            "recall_top20": float(recall_top20),
            "precision_top20": float(precision_top20),
            "label_censurado": bool(n_censurado > 0),
            # Lo que negocio necesita saber: en el top N%, ¿lo prometido
            # coincide con lo ocurrido? Ver docs/07_Informe_calibracion.md.
            "confiabilidad_tramo_alto": confiabilidad,
        }
    else:
        print(f"  Sin métricas de test: {len(test):,} filas con una sola clase "
              f"(el snapshot {test['snapshot_ts'].max():%Y-%m} todavía no tiene desenlace).")

    # === Punto de corte (cruce Sens = Espec) — artefacto del modelo ===
    # Se calcula sobre el test out-of-time: es la muestra que mejor representa
    # cómo se comportará el modelo en producción. En modo scoring el test no
    # tiene desenlace, así que se usa el validation set temporal.
    if PUEDE_EVALUAR:
        cutoff_info = calcular_cutoff(test["label_churn_30d"].values, y_prob)
        calculado_en = f"test (snapshots {TEST_YEAR})"
    else:
        cutoff_info = calcular_cutoff(y_val, prob_val_cal)
        calculado_en = "validation set temporal (ultimos 2 meses del train) — modo scoring"
    cutoff_path = guardar_cutoff(cutoff_info, OUT, MODEL_PATH, calculado_en,
                                 extra={"cutoff_date_entrenamiento": str(CUTOFF.date())})
    cutoff_info["calculado_en"] = calculado_en
    imprimir_cutoff(cutoff_info, cutoff_path)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    results = {
        "unidad_observacion": "donante (contact_id) x snapshot mensual",
        "regla_label": "1 si TODOS los pledges activos del donante cancelan en (t, t+30d] "
                       "(= min del label pledge-level dentro del grupo)",
        "n_train": len(train),
        "n_test": len(test),
        "tasa_churn_train": float(train["label_churn_30d"].mean()),
        "tasa_churn_test": float(test["label_churn_30d"].mean()),
        "donantes_overlap_train_test": overlap,
        "donantes_overlap_pct_test": 100 * overlap / len(test_donantes),
        "corte": {
            "cutoff_date": str(CUTOFF.date()),
            "test_year": TEST_YEAR,
            "train_years": f"<= {TRAIN_END}",
            "snapshots_dataset": [f"{snap_min:%Y-%m}", f"{snap_max:%Y-%m}"],
            "snapshots_train": [f"{train['snapshot_ts'].min():%Y-%m}",
                                f"{train['snapshot_ts'].max():%Y-%m}"],
            "snapshots_test": [f"{test['snapshot_ts'].min():%Y-%m}",
                               f"{test['snapshot_ts'].max():%Y-%m}"],
            "modo": MODO,
            "n_test_label_censurado": n_censurado,
            "pct_test_label_censurado": 100 * n_censurado / len(test),
        },
        "config": {
            "auto_class_weights": None,
            "learning_rate": 0.03,
            "iterations": 800,
            "best_iteration": int(model.get_best_iteration()),
            "orden": "contact_id, snapshot_ts",
            "boost_from_average": True,
            "validacion": "temporal (ultimos 2 meses del train)",
            "anti_leakage_validado": "parcial (monotonicidad = diagnóstico, ver docstring)",
            "calibracion": "IsotonicRegression",
            "features": features,
        },
        "catboost": metricas_test,
        "calibracion": {
            "metodo": "IsotonicRegression",
            "entrenado_en": "validation set temporal (ultimos 2 meses del train)",
            "archivo": "motor2_donante_calibrador.pkl",
            "ece_val_antes": float(ece_val_antes),
            "ece_val_despues": float(ece_val_despues),
            "mejora_ece_val": float(ece_val_antes - ece_val_despues),
        },
        "baselines": baselines,
        "punto_corte": {**cutoff_info, "archivo": cutoff_path.name},
        "advertencia_comparabilidad": (
            "Las métricas NO son comparables con las de results/motor2 (pledge-level): "
            "el label de donante tiene menor prevalencia, así que la PR-AUC baja "
            "mecánicamente aunque el modelo sea igual de bueno."),
        "feature_importance": importance.to_dict(orient="records"),
    }
    with open(OUT / "resultados_motor2_donante.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    importance.to_csv(OUT / "feature_importance_motor2_donante.csv", index=False)

    test_out = test[["contact_id", "snapshot_ts", "label_churn_30d",
                     "n_pledges_activos"]].copy()
    test_out["prob_churn_donante"] = y_prob.astype("float32")
    test_out["prob_churn_donante_cruda"] = y_prob_cruda.astype("float32")
    test_out["alerta"] = aplicar_cutoff(y_prob, cutoff_info["cutoff"])
    test_out = test_out.sort_values("prob_churn_donante", ascending=False).reset_index(drop=True)
    test_out.to_parquet(OUT / "test_predictions_motor2_donante.parquet", index=False)
    test_out.to_csv(OUT / "test_predictions_motor2_donante.csv", index=False)

    print(f"\n  Corte: {CUTOFF.date()}  |  train <= {TRAIN_END}  |  test {TEST_YEAR}  |  modo: {MODO}")
    if PUEDE_EVALUAR:
        print(f"  CatBoost AUC (calibrado):    {auc:.4f}")
        for k, v in baselines.items():
            print(f"  {k:24s} AUC={v['auc']:.4f}   lift={auc - v['auc']:+.4f}")
        print(f"  PR-AUC:                      {pr_auc:.4f}  "
              f"(tasa base {test['label_churn_30d'].mean():.4f})")
        print(f"  Brier crudo: {brier_crudo:.4f}  →  calibrado: {brier:.4f}")
        print(f"  Recall @ top 20%:            {recall_top20:.4f}")
        print(f"  Precision @ top 20%:         {precision_top20:.4f}")
        print(f"  Cutoff ({METODO_CUTOFF}):             {cutoff_info['cutoff']:.4f}  "
              f"(sens {cutoff_info['sensibilidad']:.1%} / espec "
              f"{cutoff_info['especificidad']:.1%}, {cutoff_info['pct_donantes_en_alerta']:.1%} en alerta)")
    else:
        print("  Métricas de test: N/A (modo scoring)")
    print(f"  Predicciones escritas: {len(test_out):,} filas")
    print(f"  Top features:\n{importance.head(8).to_string(index=False)}")
    print(f"\n[OK] Artefactos en {OUT}")


if __name__ == "__main__":
    main()
