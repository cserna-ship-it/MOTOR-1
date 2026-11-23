#!/usr/bin/env python3
"""
01_train_motor1.py — Entrenamiento Motor 1 (v8)

PROPÓSITO
---------
Entrenar el modelo de PROBABILIDAD DE ÉXITO DE COBRO (P1) con CatBoost.
Cada intento de cobro recibe una probabilidad entre 0 y 1 de que sea
exitoso (`status == "Paid"`).

LA VERSIÓN v4 vs v3
-------------------
v4 agrega 14 features de Allocation (downgrades, method changes, etc.)
sobre el conjunto de features de v3. Las features v3 son las "core" del
modelo; las features alloc aportan ~0.005 de AUC de lift en pruebas
(modesto pero consistente). La comparación contra v3 está en los
resultados (`lift_v4_vs_v3`).

VARIABLE OBJETIVO
-----------------
`is_success`: 1 si el intento fue cobrado (status="Paid"), 0 si no.
Definida en 01_create_staging.py:85.

SPLIT TEMPORAL OUT-OF-TIME
--------------------------
- Train: intentos con attempt_ts ∈ [2022, 2024]
- Test:  intentos con attempt_ts ∈ 2025

Es un split CRUDAMENTE temporal: el modelo se entrena solo con datos
del pasado y se evalúa con datos del futuro. Esto simula el escenario
real de producción donde se predice sobre intentos que aún no han
ocurrido.

NO se hace split aleatorio (cross-validation random), porque eso
introduciría leakage temporal: la racha de fallos del intento 100 del
mismo pledge contiene información parcial del intento 95.

SAMPLING DEL TRAIN (v8)
-----------------------
Se usa `sample_frac=1.0` — se entrena con el 100% del train (2022-2024).
En versiones previas se usaba 0.25 por límites de RAM; tras acordarlo con
el cliente se entrena con todos los datos disponibles.

BALANCEO DE CLASES (v8)
-----------------------
Se usa `auto_class_weights="Balanced"`. El target está desbalanceado
(~85% éxito). El balanceo automático pondera inversamente a la frecuencia
de cada clase, dándole más peso a los fallos (clase minoritaria). En pruebas
el AUC resultó levemente inferior que sin balanceo, pero tras discutirlo con
el cliente se prioriza el balanceo (mejor sensibilidad sobre la clase de
fallos, que es la accionable para el negocio).

ORDEN Y ANTI-LEAKAGE (v8)
-------------------------
Aunque el orden cronológico por pledge y el anti-leakage (shift(1)) ya se
garantizan aguas arriba (01_create_staging.py), este script:
  1. RE-ORDENA explícitamente por (pledge_id, attempt_ts, payment_id) tras
     cargar cada slice — necesario porque `sample()` de pandas devuelve las
     filas en orden aleatorio, lo que rompería el holdout de validación.
  2. VALIDA la integridad temporal (validate_temporal_integrity): que no
     existan `days_since_last_attempt` negativos y que el primer intento de
     cada pledge llegue con historial acumulado en cero — evidencia de que
     el shift(1) upstream está intacto. Si la validación falla, aborta.

HIPERPARÁMETROS DEL MODELO
--------------------------
- 180 iteraciones (árboles)
- profundidad 6
- learning_rate 0.10
- early stopping con paciencia 15 iteraciones
- 10% del train se reserva para validación interna (early stopping)
- auto_class_weights="Balanced"

MÉTRICAS REPORTADAS
-------------------
- AUC, PR-AUC, Brier
- Classification report + matriz de confusión @ threshold=0.5
- AUC por subset (con vs sin allocations) — útil para entender el
  aporte real de las features alloc
- Lift vs baseline trivial (lookup por pre_failure_streak)
- Lift vs v3 (AUC referencia 0.9074)
- Feature importance

ENTRADA / SALIDA
----------------
Lee:    data/features/motor1_features.parquet
        data/features/allocation_features.parquet
Genera: results/motor1/
            motor1_catboost.cbm                 → modelo serializado
            resultados_motor1.json              → métricas
            feature_importance.csv              → top features
            test_predictions_motor1.parquet     → predicciones (para Motor 3)

NOTA SOBRE RE-ENTRENAMIENTO
---------------------------
Si llegan datos nuevos del cliente (ej: 2026), antes de re-correr este
script hay que decidir el nuevo split temporal:
    - Opción A: train=2022-2025, test=2026 (mover ventana)
    - Opción B: train=2025 reciente, test=2026 (ventana móvil)
Editar las llamadas a load_slice() abajo.
"""
from pathlib import Path
import sys
import json
import gc
import pickle
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/ al path
from config import get_cutoff_date
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, average_precision_score,
    classification_report, confusion_matrix
)
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
MOTOR1_FEATURES = ROOT / "data/features/motor1_features.parquet"
ALLOC_FEAT = ROOT / "data/features/allocation_features.parquet"
OUT = ROOT / "results/motor1"
OUT.mkdir(parents=True, exist_ok=True)

# Clave de orden canónica. El mismo orden que garantiza 01_create_staging.py
# antes de calcular las features de secuencia con shift(1). Se re-aplica en
# load_slice() para blindar el anti-leakage dentro de este propio script.
SORT_KEYS = ["pledge_id", "attempt_ts", "payment_id"]

# === Features v3 (núcleo del modelo, 16 features) ===
# Numéricas — todas derivadas en los pasos de feature engineering:
#   pre_failure_streak       → racha de fallos previos (predictor #1)
#   days_since_last_attempt  → tiempo desde el intento anterior
#   success/failure_count_cum → conteos acumulados históricos
#   payment_amount           → monto del intento (COP)
#   month/day_of_month/day_of_week → componentes de calendario
#                             (NO se pasa attempt_ts directo al modelo)
#   contact_age_days         → antigüedad del registro del donante en SF
FEAT_NUM_V3 = ["pre_failure_streak", "days_since_last_attempt",
               "success_count_cum", "failure_count_cum", "payment_amount",
               "month", "day_of_month", "day_of_week", "contact_age_days"]

# Categóricas — CatBoost las maneja nativamente sin one-hot.
# v8: se usa `payment_method` en lugar de `payment_submethod`. El cliente
# reportó que payment_submethod está frecuentemente vacío, mientras que
# payment_method está poblado en ~99.999% de los intentos y con categorías
# limpias (Auto Debit (ACH), Credit Card, Debit Card, eWallet, ...).
FEAT_CAT_V3 = ["payment_gateway", "payment_method", "bank_canonical",
               "state_norm", "gender_norm", "age_band", "ses_level"]

# === Features nuevas en v4 (Allocation, 14 features) ===
# Cuántos cambios contractuales tuvo el pledge (conteos + recencia + flags 90d).
FEAT_NUM_ALLOC = ["alloc_n_total", "alloc_n_downgrades_total",
                  "alloc_n_method_changes_total", "alloc_n_day_changes_total",
                  "alloc_n_frequency_changes_total", "alloc_n_errors_total",
                  "alloc_days_since_last",
                  "alloc_recent90_downgrade", "alloc_recent90_method_change",
                  "alloc_recent90_day_change", "alloc_recent90_frequency_change",
                  "alloc_recent90_error"]
FEAT_CAT_ALLOC = ["alloc_last_type", "alloc_last_status"]

FEAT_NUM = FEAT_NUM_V3 + FEAT_NUM_ALLOC
FEAT_CAT = FEAT_CAT_V3 + FEAT_CAT_ALLOC
FEATURES = FEAT_NUM + FEAT_CAT


def load_slice(year_min, year_max, sample_frac=None):
    """
    Carga un slice temporal del dataset (intentos cuyo attempt_ts está en
    el rango [year_min, year_max]) y lo enriquece con allocation features.

    Args:
        year_min, year_max: rango de años (inclusivo, sobre attempt_ts.dt.year)
        sample_frac: si no None, samplea ese fraccion del slice (para train).
                     Random seed fijo (42) para reproducibilidad.

    Por qué lee por batches:
        motor1_features tiene 5.3M filas. Cargarlo entero consumiría >4GB
        de RAM. La lectura por batches de 300k filas con filtrado al vuelo
        permite trabajar con la slice deseada sin saturar memoria.

    ORDEN (v8):
        Tras concatenar los batches se RE-ORDENA por (pledge_id, attempt_ts,
        payment_id). Es imprescindible: `df.sample(frac=...)` devuelve las
        filas en orden aleatorio, y el holdout de validación de early stopping
        toma el último 10% del train asumiendo un orden determinista por
        pledge. Sin este re-orden, el holdout sería aleatorio y no
        reproducible entre corridas.

    Pasos de limpieza/tipos:
        - is_success → int8 (target binario)
        - payment_amount → float32 después de quitar caracteres no numéricos
          (algunos registros tienen "$" o "," en el monto)
        - contact_age_days → float32
        - month/day_of_month/day_of_week → derivadas de attempt_ts (int8)
        - features alloc numéricas → fillna(0), excepto days_since_last
          que va a 99999 (sin eventos)
        - categóricas → fillna("UNKNOWN") + cast a str
    """
    cols_v3 = ["payment_id", "pledge_id", "attempt_ts", "is_success", "payment_amount",
               "payment_gateway", "payment_method",
               "days_since_last_attempt", "pre_failure_streak",
               "success_count_cum", "failure_count_cum",
               "bank_canonical", "state_norm", "gender_norm",
               "age_band", "ses_level", "contact_age_days"]
    pf = pq.ParquetFile(MOTOR1_FEATURES)
    chunks = []
    for batch in pf.iter_batches(columns=cols_v3, batch_size=300_000):
        c = batch.to_pandas()
        c = c[(c["attempt_ts"].dt.year >= year_min) & (c["attempt_ts"].dt.year <= year_max)]
        if len(c) == 0:
            continue
        if sample_frac is not None:
            c = c.sample(frac=sample_frac, random_state=42)
        chunks.append(c)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    # Sin filas no hay ni columnas, y el merge de abajo revienta con un
    # KeyError('payment_id') que no dice nada. Motor 1 se entrena y evalúa sobre
    # INTENTOS DE COBRO: si el año pedido no tiene intentos cargados, no hay nada
    # que hacer y hay que decirlo claro.
    if len(df) == 0:
        raise SystemExit(
            f"\n[ERROR] No hay intentos de cobro entre {year_min} y {year_max} en\n"
            f"        {MOTOR1_FEATURES}\n\n"
            f"        Motor 1 usa el AÑO DE LA FECHA DE CORTE como año de test. Si el\n"
            f"        corte cae en un año sin pagos cargados (p.ej. --cutoff-date\n"
            f"        2026-01-03 con pagos hasta 2025-12-31), no hay test posible.\n\n"
            f"        Opciones:\n"
            f"          a) usar un corte dentro del último año CON pagos (2025-12-31), o\n"
            f"          b) cargar los pagos del año nuevo en raw/ y re-correr\n"
            f"             00_unify_raw_data.py + 01_create_staging.py\n\n"
            f"        Nota: Motor 2 SÍ admite el corte 2026-01-03 — su unidad es el\n"
            f"        snapshot mensual, no el intento de cobro.\n")

    # === Re-orden canónico por pledge (v8, anti-shuffle de sample) ===
    # Garantiza orden determinista (pledge_id, attempt_ts, payment_id) — el
    # mismo que usa el pipeline upstream para el shift(1). Deshace el barajado
    # introducido por df.sample() y hace reproducible el holdout de validación.
    if len(df):
        df = df.sort_values(SORT_KEYS).reset_index(drop=True)

    # Merge con allocation features
    af = pd.read_parquet(ALLOC_FEAT)
    df = df.merge(af, on="payment_id", how="left")
    del af
    gc.collect()

    # Tipos
    df["is_success"] = df["is_success"].astype("int8")
    df["payment_amount"] = pd.to_numeric(df["payment_amount"].astype(str).str.replace(r"[^\d\.]","",regex=True),
                                          errors="coerce").fillna(0).astype("float32")
    df["contact_age_days"] = pd.to_numeric(df["contact_age_days"], errors="coerce").astype("float32")
    df["month"] = df["attempt_ts"].dt.month.astype("int8")
    df["day_of_month"] = df["attempt_ts"].dt.day.astype("int8")
    df["day_of_week"] = df["attempt_ts"].dt.dayofweek.astype("int8")
    for c in ["pre_failure_streak", "success_count_cum", "failure_count_cum", "days_since_last_attempt"]:
        df[c] = df[c].astype("float32")
    # Imputar allocation features para filas sin match (no debería)
    for c in FEAT_NUM_ALLOC:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0 if "days" not in c else 99999).astype("float32")
    for c in FEAT_CAT:
        df[c] = df[c].astype(str).fillna("UNKNOWN")
    return df


def validate_temporal_integrity(df, nombre, full_history=False):
    """
    Blindaje anti-leakage DENTRO de este script (v8).

    El shift(1) y el orden cronológico por pledge ya se garantizan aguas
    arriba (01_create_staging.py:add_sequence_features), pero el cliente
    solicitó validarlo también acá. Comprueba invariantes que, juntas,
    evidencian que las features de secuencia solo usan información
    estrictamente anterior a cada intento.

    Invariantes que se validan en CUALQUIER slice (train o test):
      1. `days_since_last_attempt` nunca es negativo. Un valor negativo
         significaría que un intento con fecha posterior aparece antes que
         uno anterior del mismo pledge → orden roto → leakage potencial.
      2. Los conteos acumulados (success_count_cum, failure_count_cum) son
         MONÓTONOS NO DECRECIENTES dentro de cada pledge. Un decremento al
         avanzar en el tiempo significaría que una fila "ve" menos historial
         que una posterior → orden roto.

    Invariante adicional SOLO cuando `full_history=True` (el slice contiene el
    inicio real del pledge, p.ej. train 2022-2024, que es el comienzo de los
    datos):
      3. El PRIMER intento de cada pledge llega con historial acumulado en
         cero. Es la firma directa del shift(1): el intento #1 no puede "ver"
         ningún intento previo.
      (En un slice que NO contiene el inicio del pledge —p.ej. test 2025— el
       "primer intento del slice" trae legítimamente historial de años previos,
       así que esta comprobación no aplica y se omite.)

    Aborta con AssertionError si alguna invariante no se cumple.
    Requiere que `df` esté ordenado por (pledge_id, attempt_ts, payment_id).
    """
    neg = int((df["days_since_last_attempt"].dropna() < 0).sum())
    assert neg == 0, (
        f"[{nombre}] {neg} filas con days_since_last_attempt negativo → "
        f"orden cronológico roto (posible leakage). Revisar el orden por pledge.")

    for col in ["success_count_cum", "failure_count_cum"]:
        dec = int((df.groupby("pledge_id", sort=False)[col].diff() < 0).sum())
        assert dec == 0, (
            f"[{nombre}] {dec} filas donde {col} decrece dentro del pledge → "
            f"orden cronológico roto (posible leakage).")

    extra = ""
    if full_history:
        first = df.groupby("pledge_id", sort=False).head(1)
        bad = int(((first["success_count_cum"] != 0) |
                   (first["failure_count_cum"] != 0) |
                   (first["pre_failure_streak"] != 0)).sum())
        assert bad == 0, (
            f"[{nombre}] {bad} pledges cuyo PRIMER intento ya trae historial "
            f"acumulado ≠ 0 → el shift(1) upstream no está intacto (leakage).")
        extra = " + shift(1) verificado en el primer intento"

    print(f"  [OK anti-leakage] {nombre}: days≥0 y conteos monótonos{extra} "
          f"({df['pledge_id'].nunique():,} pledges validados).")


def main():
    CUTOFF = get_cutoff_date()
    TEST_YEAR  = CUTOFF.year           # año de test = año del corte
    TRAIN_END  = TEST_YEAR - 1         # último año de train
    TRAIN_START = 2022                 # inicio histórico fijo de los datos

    # === Train: años 2022 → (TEST_YEAR - 1) ===
    # sample_frac=1.0 → 100% del train (v8; acordado con el cliente).
    print(f"[1/4] Cargando train (años {TRAIN_START}-{TRAIN_END}, sample_frac=1.0)...")
    train = load_slice(TRAIN_START, TRAIN_END, sample_frac=1.00)
    print(f"  Train: {len(train):,}")

    # Blindaje anti-leakage (v8): valida orden y shift(1) en el propio script.
    # full_history=True: 2022 es el inicio de los datos, así que el primer
    # intento de cada pledge en el train es su primer intento real.
    validate_temporal_integrity(train, f"train {TRAIN_START}-{TRAIN_END}", full_history=True)

    print("[2/4] Entrenando CatBoost con Allocation features (auto_class_weights)...")
    # Validación interna: el último 10% de filas (orden determinista por
    # (pledge_id, attempt_ts, payment_id) garantizado por load_slice) se usa
    # como validation set para early stopping. NO es el test final — el test
    # es el slice de 2025.
    val_size = int(len(train) * 0.1)
    tr_idx = train.iloc[:-val_size].index
    val_idx = train.iloc[-val_size:].index

    # CatBoost — hiperparámetros conservadores y robustos:
    #   iterations=180        ← número máximo de árboles (early stopping puede
    #                           cortar antes)
    #   depth=6               ← profundidad estándar para tabular
    #   learning_rate=0.10    ← convergencia rápida
    #   loss_function=Logloss ← apropiado para clasificación binaria
    #   eval_metric=AUC       ← se monitorea AUC en validación
    #   od_type=Iter, od_wait=15 → para early stopping: si no mejora en
    #                               15 iteraciones, corta. Previene overfitting.
    #   auto_class_weights="Balanced" (v8) → pondera inversamente a la
    #                               frecuencia de clase. El target está
    #                               desbalanceado (~85% éxito); esto da más
    #                               peso a los fallos (clase accionable).
    model = CatBoostClassifier(
        iterations=180, depth=6, learning_rate=0.10,
        loss_function="Logloss", eval_metric="AUC",
        cat_features=FEAT_CAT, random_seed=42,
        verbose=30, thread_count=-1,
        od_type="Iter", od_wait=15,
        auto_class_weights="Balanced",
    )
    model.fit(
        train.loc[tr_idx, FEATURES], train.loc[tr_idx, "is_success"],
        eval_set=(train.loc[val_idx, FEATURES], train.loc[val_idx, "is_success"]),
        use_best_model=True,
    )
    model.save_model(str(OUT / "motor1_catboost.cbm"))

    # === Calibración isotónica (v9) ===
    # El validation set del early stopping se reutiliza para entrenar el
    # calibrador. IsotonicRegression aprende a corregir el sesgo que introduce
    # auto_class_weights="Balanced" en las probabilidades crudas, y también
    # actúa como seguro ante prior shift (si en el futuro el ratio de clases
    # cambia, se recalibra este objeto sin reentrenar el modelo grande).
    #
    # Por qué isotónica y no Platt (sigmoid)?
    #   - Platt asume que el sesgo es monotónico y simétrico.
    #   - Isotónica es no paramétrica: aprende cualquier forma de corrección
    #     mientras sea monótona. Más flexible para datos de este tamaño.
    #   - Con >200k filas en val el riesgo de overfitting de isotónica es mínimo.
    print("  [calibración] Entrenando IsotonicRegression sobre validation set...")
    prob_val_cruda = model.predict_proba(train.loc[val_idx, FEATURES])[:, 1]
    y_val = train.loc[val_idx, "is_success"].values

    calibrador = IsotonicRegression(out_of_bounds="clip")
    calibrador.fit(prob_val_cruda, y_val)

    # Guardar calibrador junto al modelo
    with open(OUT / "motor1_calibrador.pkl", "wb") as f:
        pickle.dump(calibrador, f)

    # ECE en validation: antes vs después de calibrar
    def ece(probs, labels, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        total = len(probs)
        err = 0.0
        for i in range(n_bins):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            err += mask.sum() / total * abs(probs[mask].mean() - labels[mask].mean())
        return err

    prob_val_cal = calibrador.transform(prob_val_cruda)
    ece_antes = ece(prob_val_cruda, y_val)
    ece_despues = ece(prob_val_cal, y_val)
    print(f"  [calibración] ECE val — antes: {ece_antes:.4f}  →  después: {ece_despues:.4f}  "
          f"(mejora: {ece_antes - ece_despues:+.4f})")

    train_l = train[["pre_failure_streak", "is_success"]].copy()
    train_l["streak_cap"] = train_l["pre_failure_streak"].clip(upper=20).astype(int)
    lookup_streak = train_l.groupby("streak_cap")["is_success"].mean().to_dict()
    train_mean = float(train["is_success"].mean())

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False)

    n_train = len(train)
    tasa_train = float(train["is_success"].mean())
    del train, train_l
    gc.collect()

    print(f"[3/4] Cargando test {TEST_YEAR}...")
    test = load_slice(TEST_YEAR, TEST_YEAR)
    print(f"  Test: {len(test):,}")
    validate_temporal_integrity(test, f"test {TEST_YEAR}")

    y_prob_cruda = model.predict_proba(test[FEATURES])[:, 1]
    y_prob = calibrador.transform(y_prob_cruda)   # probabilidades calibradas
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(test["is_success"], y_prob)
    pr_auc = average_precision_score(test["is_success"], y_prob)
    brier = brier_score_loss(test["is_success"], y_prob)
    brier_crudo = brier_score_loss(test["is_success"], y_prob_cruda)
    cls_report = classification_report(test["is_success"], y_pred, output_dict=True)
    cm = confusion_matrix(test["is_success"], y_pred).tolist()

    # ECE en test: antes vs después
    ece_test_antes  = ece(y_prob_cruda, test["is_success"].values)
    ece_test_despues = ece(y_prob,      test["is_success"].values)
    print(f"  [calibración] ECE test — antes: {ece_test_antes:.4f}  →  después: {ece_test_despues:.4f}")

    p_baseline = test["pre_failure_streak"].clip(upper=20).astype(int).map(lookup_streak).fillna(train_mean).values.astype(float)
    auc_baseline = roc_auc_score(test["is_success"], p_baseline)

    # === Análisis de lift por subset ===
    # Calculamos AUC separadamente para:
    #   - intentos de pledges CON historial de allocations (alloc_n_total > 0)
    #   - intentos de pledges SIN historial (alloc_n_total == 0)
    # Si la diferencia es grande, las features alloc están aportando.
    # Si es pequeña, su aporte es marginal.
    has_alloc = test["alloc_n_total"] > 0
    auc_with_alloc = roc_auc_score(test.loc[has_alloc, "is_success"], y_prob[has_alloc])
    auc_no_alloc = roc_auc_score(test.loc[~has_alloc, "is_success"], y_prob[~has_alloc])

    results = {
        "n_train": n_train, "n_test": len(test),
        "tasa_exito_train": tasa_train,
        "tasa_exito_test": float(test["is_success"].mean()),
        "config_v8": {
            "sample_frac": 1.0,
            "auto_class_weights": "Balanced",
            "payment_field": "payment_method (reemplaza payment_submethod)",
            "orden": "pledge_id, attempt_ts, payment_id",
            "anti_leakage_validado": True,
        },
        "catboost": {
            "auc": float(auc), "pr_auc": float(pr_auc),
            "brier_calibrado": float(brier), "brier_crudo": float(brier_crudo),
            "ece_test_antes_calibrar": float(ece_test_antes),
            "ece_test_despues_calibrar": float(ece_test_despues),
            "classification_report": cls_report, "confusion_matrix": cm,
        },
        "calibracion": {
            "metodo": "IsotonicRegression",
            "entrenado_en": "validation set (ultimo 10% del train)",
            "archivo": "motor1_calibrador.pkl",
            "ece_val_antes":   float(ece_antes),
            "ece_val_despues": float(ece_despues),
            "mejora_ece_val":  float(ece_antes - ece_despues),
        },
        "auc_subset_with_alloc": float(auc_with_alloc),
        "auc_subset_no_alloc": float(auc_no_alloc),
        "n_with_alloc": int(has_alloc.sum()),
        "n_no_alloc": int((~has_alloc).sum()),
        "baseline_B1_lookup_streak": {"auc": float(auc_baseline)},
        "lift_auc_vs_baseline": float(auc - auc_baseline),
        "auc_v3_referencia": 0.9074,
        "lift_vs_v3": float(auc - 0.9074),
        "feature_importance": importance.to_dict(orient="records"),
        "features_used": FEATURES,
        "features_alloc": FEAT_NUM_ALLOC + FEAT_CAT_ALLOC,
    }
    with open(OUT / "resultados_motor1.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    test_out = test[["payment_id", "pledge_id", "attempt_ts", "is_success", "payment_amount"]].copy()
    test_out["prob_success"]       = y_prob.astype("float32")        # calibrada ← usar esta
    test_out["prob_success_cruda"] = y_prob_cruda.astype("float32")  # cruda     ← solo referencia
    test_out.to_parquet(OUT / "test_predictions_motor1.parquet", index=False)

    print(f"\n[4/4] RESULTADOS Motor 1 (con Allocation, auto_class_weights):")
    print(f"  AUC:                       {auc:.4f}")
    print(f"  AUC v3 (referencia):       0.9074")
    print(f"  Lift vs v3:                {auc - 0.9074:+.4f}")
    print(f"  AUC baseline (lookup):     {auc_baseline:.4f}")
    print(f"  Lift vs baseline:          {auc - auc_baseline:+.4f}")
    print(f"  PR-AUC: {pr_auc:.4f} | Brier crudo: {brier_crudo:.4f} → calibrado: {brier:.4f}")
    print(f"  ECE test  — antes: {ece_test_antes:.4f}  →  después: {ece_test_despues:.4f}")
    print(f"  AUC subset CON allocations: {auc_with_alloc:.4f} (n={has_alloc.sum():,})")
    print(f"  AUC subset SIN allocations: {auc_no_alloc:.4f} (n={(~has_alloc).sum():,})")
    print(f"  Top 12 features:")
    for _, r in importance.head(12).iterrows():
        marker = " 🆕" if r["feature"] in (FEAT_NUM_ALLOC + FEAT_CAT_ALLOC) else ""
        print(f"    {r['feature']:<35} {r['importance']:.2f}%{marker}")


if __name__ == "__main__":
    main()
