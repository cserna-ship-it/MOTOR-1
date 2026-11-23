#!/usr/bin/env python3
"""
motor2_donante_common.py — Lógica compartida del Motor 2 a nivel DONANTE

POR QUÉ EXISTE ESTE MÓDULO
--------------------------
La agregación pledge → donante tiene que ser BIT A BIT LA MISMA en los tres
puntos donde se usa:

    01b_motor2_dataset_donante.py   construye el dataset de entrenamiento
    02_train_motor2_donante.py      entrena sobre ese dataset
    04_score_motor2_donante.py      agrega el snapshot mensual y puntúa

Si el snapshot mensual se agregara (o se imputara) con una regla distinta a la
que vio el modelo en entrenamiento, las predicciones quedarían sesgadas sin
ningún error visible. La bitácora del cambio (docs/Cambios_Motor_2.pdf,
sección "Nota de diseño") señalaba esta duplicación como deuda a saldar antes
de producción: este módulo la salda.

También vive acá el punto de corte (cruce sensibilidad = especificidad) que convierte la probabilidad en
la bandera `alerta`: se calcula en 02b, se guarda como motor2_donante_cutoff.json
y 04b lo lee. Ver la sección "PUNTO DE CORTE" al final del módulo.

Los scripts que lo importan hacen:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/
    from motor2_donante_common import (
        FEATURES_DONANTE, adjuntar_contact_id, agregar_a_donante,
        preparar_features_donante,
    )

REGLA DE NEGOCIO DEL LABEL
--------------------------
Un donante abandona en (t, t+30d] si y solo si TODOS sus pledges activos en el
snapshot t cancelan dentro de esa ventana:

    label_donante(d, t) = 1  ⟺  label_churn_30d(p, t) = 1  ∀ p ∈ P(d, t)

Como el label pledge-level ya es binario, "para todo p vale 1" es el MÍNIMO del
grupo (contact_id, snapshot_ts).
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Features que consume el modelo de donante, en orden canónico. La lista vive
# acá (y no en el script de entrenamiento) para que el script de puntuación
# pueda validar contra la MISMA fuente de verdad.
FEATURES_DONANTE = [
    "n_pledges_activos",
    "monto_total_mensual",
    "streak_max",
    "streak_promedio",
    "tenure_max",
    "tenure_min",
    "cum_intentos_total",
    "cum_exitos_total",
    "cum_fallos_total",
    "n_intentos_30d_total",
    "n_fallos_30d_total",
    "days_since_last_attempt_min",
    "n_bancos_distintos",
    "n_metodos_distintos",
    "success_rate_hist_pond",
]

# Sin categóricas a nivel donante: last_bank_name / last_payment_method son de
# UN pledge, no del donante, y se reemplazaron por conteos de diversidad.
CAT_FEATURES_DONANTE = []

# Sentinel de "sin actividad previa conocida" — el mismo valor y el mismo
# significado que usa el pipeline pledge-level (02_train_motor2.py).
SENTINEL_SIN_ACTIVIDAD = 999


def resolver_pledge_to_contact(staging_path):
    """
    Mapeo pledge_id -> contact_id leído del staging, resuelto contra
    inconsistencias.

    Un pledge debería tener un único contact_id en todo su historial de
    intentos. Si el CRM tiene algún caso con más de uno (error de captura, o un
    pledge reasignado), se toma el MÁS FRECUENTE en el historial de intentos de
    ese pledge. Es la misma regla que aplica 01_motor2_dataset.py al escribir la
    columna contact_id, para que ambos caminos den el mismo resultado.
    """
    stg = pd.read_parquet(staging_path, columns=["pledge_id", "contact_id"])
    conteo = (stg.groupby(["pledge_id", "contact_id"]).size()
              .reset_index(name="n_intentos_con_ese_contacto"))
    resuelto = (conteo.sort_values("n_intentos_con_ese_contacto", ascending=False)
                .drop_duplicates(subset=["pledge_id"], keep="first")
                [["pledge_id", "contact_id"]])
    n_total = conteo["pledge_id"].nunique()
    n_amb = int((conteo.groupby("pledge_id")["contact_id"].nunique() > 1).sum())
    if n_amb:
        print(f"  [AVISO] {n_amb:,} de {n_total:,} pledges tenían más de un "
              f"contact_id — resuelto tomando el más frecuente.")
    return resuelto


def adjuntar_contact_id(pledge_df, staging_path=None):
    """
    Garantiza que `pledge_df` tenga una columna contact_id utilizable.

    Desde el cambio a nivel donante, 01_motor2_dataset.py YA escribe contact_id
    en motor2_pledge_month.parquet. Si la columna está, se usa tal cual — hacer
    otro merge contra el staging generaría contact_id_x / contact_id_y y
    rompería todo aguas abajo con un KeyError.

    El camino de fallback (releer el staging) queda para poder procesar parquets
    generados con la versión ANTERIOR del script, que no traían la columna.

    Las filas sin contact_id se descartan: no se puede agrupar por donante lo
    que no se puede atribuir a un donante.
    """
    if "contact_id" in pledge_df.columns:
        print("  contact_id ya viene en el dataset pledge-level (no se relee el staging).")
        df = pledge_df
    else:
        if staging_path is None or not Path(staging_path).exists():
            raise SystemExit(
                "\n[ERROR] El dataset pledge-level no tiene contact_id y no hay staging "
                "para resolverlo.\n"
                "        Regenera las features con la versión actual de "
                "01_motor2_dataset.py.\n")
        print("  contact_id ausente en el dataset pledge-level — se resuelve desde el staging.")
        pledge_to_contact = resolver_pledge_to_contact(staging_path)
        df = pledge_df.merge(pledge_to_contact, on="pledge_id", how="left")

    sin_contacto = int(df["contact_id"].isna().sum())
    if sin_contacto:
        pct = 100 * sin_contacto / len(df)
        print(f"  [AVISO] {sin_contacto:,} filas ({pct:.2f}%) sin contact_id — se EXCLUYEN "
              f"(pledges sin intentos en el staging).")
        df = df.dropna(subset=["contact_id"]).copy()
    return df


def agregar_a_donante(df, con_label=True):
    """
    Agrega el dataset pledge × snapshot a (contact_id, snapshot_ts).

    Cada feature usa la operación que tiene sentido de negocio, no un promedio
    uniforme (ver tabla en la bitácora del cambio):

        n_pledges_activos           nunique  — cuántos compromisos tiene hoy
        monto_total_mensual         sum      — valor económico del donante
        streak_max                  max      — la peor racha domina el riesgo
        tenure_max / tenure_min     max/min  — pledge más viejo / más nuevo
        cum_*_total                 sum      — track record combinado
        success_rate_hist_pond      ponderado por intentos de cada pledge
        days_since_last_attempt_min min      — actividad más reciente en
                                               CUALQUIERA de sus pledges
        n_bancos / n_metodos        nunique  — diversidad de medios de pago

    `con_label=False` para el snapshot mensual de producción: el mes en curso
    todavía no tiene desenlace, así que no hay label_churn_30d que agregar.
    """
    llaves = ["contact_id", "snapshot_ts"]
    agg_kwargs = dict(
        n_pledges_activos=("pledge_id", "nunique"),
        monto_total_mensual=("last_payment_amount", "sum"),
        streak_max=("streak_actual", "max"),
        streak_promedio=("streak_actual", "mean"),
        tenure_max=("tenure_days", "max"),
        tenure_min=("tenure_days", "min"),
        cum_intentos_total=("cum_intentos", "sum"),
        cum_exitos_total=("cum_exitos", "sum"),
        cum_fallos_total=("cum_fallos", "sum"),
        n_intentos_30d_total=("n_intentos_30d", "sum"),
        n_fallos_30d_total=("n_fallos_30d", "sum"),
        days_since_last_attempt_min=("days_since_last_attempt", "min"),
        n_bancos_distintos=("last_bank_name", "nunique"),
        n_metodos_distintos=("last_payment_method", "nunique"),
    )
    if con_label:
        # min == "todos los pledges activos cancelan en la ventana". Ver la
        # regla de negocio en el docstring del módulo.
        agg_kwargs["label_churn_30d"] = ("label_churn_30d", "min")

    agregado = df.groupby(llaves, observed=True).agg(**agg_kwargs).reset_index()

    # success_rate_hist ponderado por cum_intentos de cada pledge: un pledge con
    # 50 intentos pesa más que uno con 2. Un promedio simple entre pledges le
    # daría el mismo peso a los dos.
    tmp = df[llaves + ["success_rate_hist", "cum_intentos"]].copy()
    tmp["_exitos_para_ponderar"] = tmp["success_rate_hist"] * tmp["cum_intentos"]
    pond = tmp.groupby(llaves, observed=True).agg(
        _exitos_pond=("_exitos_para_ponderar", "sum"),
        _intentos_pond=("cum_intentos", "sum"),
    ).reset_index()
    pond["success_rate_hist_pond"] = (
        pond["_exitos_pond"] / pond["_intentos_pond"].replace(0, np.nan)
    ).fillna(0).astype("float32")

    agregado = agregado.merge(
        pond[llaves + ["success_rate_hist_pond"]], on=llaves, how="left")
    return agregado


def preparar_features_donante(df):
    """
    Imputación y tipos de las features de donante. ENTRENAMIENTO Y PUNTUACIÓN
    TIENEN QUE LLAMAR ESTO — si el snapshot mensual llegara al modelo con NaN
    donde el entrenamiento vio 999, CatBoost trataría esas filas por una rama
    distinta y las predicciones saldrían sesgadas sin lanzar ningún error.

    Modifica y devuelve una copia.
    """
    df = df.copy()

    # NaN solo si NINGÚN pledge del donante tuvo un intento previo al snapshot
    # (donante recién activado). Mismo sentinel que el pipeline pledge-level.
    df["days_since_last_attempt_min"] = (
        df["days_since_last_attempt_min"].fillna(SENTINEL_SIN_ACTIVIDAD).astype("float32"))
    df["monto_total_mensual"] = df["monto_total_mensual"].fillna(0).astype("float32")
    df["success_rate_hist_pond"] = df["success_rate_hist_pond"].fillna(0).astype("float32")

    for col in ["n_pledges_activos", "streak_max", "tenure_max", "tenure_min",
                "cum_intentos_total", "cum_exitos_total", "cum_fallos_total",
                "n_intentos_30d_total", "n_fallos_30d_total",
                "n_bancos_distintos", "n_metodos_distintos"]:
        df[col] = df[col].fillna(0).astype("int32")
    df["streak_promedio"] = df["streak_promedio"].fillna(0).astype("float32")

    faltantes = [c for c in FEATURES_DONANTE if c not in df.columns]
    if faltantes:
        raise SystemExit(
            f"\n[ERROR] Faltan features de donante en el dataframe: {faltantes}\n"
            f"        La agregación se desincronizó de FEATURES_DONANTE "
            f"(scripts/motor2_donante_common.py).\n")
    return df


# =============================================================================
#  PUNTO DE CORTE (CUTOFF) — CRUCE DE SENSIBILIDAD Y ESPECIFICIDAD
# =============================================================================
# El modelo entrega una probabilidad calibrada. Retención necesita además una
# decisión binaria: "este donante entra o no entra a la lista". Esa decisión se
# toma con un umbral que se calcula UNA vez, al entrenar, y se guarda como
# artefacto junto al modelo — exactamente igual que el calibrador. El scoring
# mensual solo lo LEE: nunca lo recalcula (el mes en curso no tiene desenlace,
# así que no habría con qué calcularlo) ni lo trae escrito a mano en el código.
#
# CRITERIO APLICADO: "cruce" — el umbral donde las curvas de sensibilidad y
# especificidad se cruzan:
#
#     τ* = argmin_τ | Sens(τ) − Espec(τ) |
#
# Es decir, el punto donde la lista detecta la misma proporción de los que se
# van que la proporción de los que se quedan que deja fuera. Se eligió por
# decisión de negocio (octubre 2026): prioriza no perder donantes, a costa de
# una lista más larga que la de Youden.
#
# REFERENCIA REGISTRADA: "youden" — τ = argmax_τ [Sens(τ) + Espec(τ) − 1].
# No se aplica, pero se guarda en el artefacto para poder comparar.
#
# Por qué NO 0,5: con una prevalencia de ~3,3 % casi ningún donante supera 0,5
# y la lista saldría prácticamente vacía. El umbral útil vive cerca de la tasa
# base, y los dos criterios lo encuentran sin tener que adivinarlo.

CUTOFF_FILENAME = "motor2_donante_cutoff.json"
METODOS_CUTOFF = ("cruce", "youden")
METODO_CUTOFF = "cruce"   # el que se aplica para la columna `alerta`


def huella_modelo(model_path):
    """SHA-256 del .cbm. Amarra el cutoff al modelo exacto con el que se calculó."""
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def _punto(y_true, prob, fpr, tpr, thr, i):
    """
    Métricas en el escalón i de la curva ROC.

    La calibración isotónica produce una función escalonada: la probabilidad
    solo toma unos cientos de valores distintos. Por eso el umbral se ubica en
    el PUNTO MEDIO entre el escalón elegido y el siguiente escalón hacia abajo:
    clasifica exactamente igual que el escalón, pero no depende de que
    float32/float64 redondeen el valor del escalón al mismo número.
    """
    escalon = float(thr[i])
    siguiente = float(thr[i + 1]) if i + 1 < len(thr) else 0.0
    cutoff = (escalon + siguiente) / 2.0

    alerta = prob >= cutoff
    vp = int((alerta & (y_true == 1)).sum())
    fp = int((alerta & (y_true == 0)).sum())
    fn = int((~alerta & (y_true == 1)).sum())
    vn = int((~alerta & (y_true == 0)).sum())
    tasa_base = float(y_true.mean())
    precision = vp / max(vp + fp, 1)
    return {
        "cutoff": cutoff,
        "prob_minima_en_alerta": escalon,
        "sensibilidad": float(tpr[i]),
        "especificidad": float(1 - fpr[i]),
        "youden_j": float(tpr[i] - fpr[i]),
        "precision": float(precision),
        "lift": float(precision / tasa_base) if tasa_base else None,
        "pct_donantes_en_alerta": float(alerta.mean()),
        "matriz_confusion": {"VP": vp, "FP": fp, "FN": fn, "VN": vn},
    }


def calcular_cutoff(y_true, prob, metodo=METODO_CUTOFF):
    """
    Punto de corte sobre probabilidades CALIBRADAS.

    metodo="cruce"  → argmin |Sens − Espec|   (el que se aplica)
    metodo="youden" → argmax  Sens + Espec − 1

    Devuelve un dict listo para serializar con el umbral del método pedido, sus
    métricas y, en `referencia_<otro>`, el punto que habría dado el otro
    criterio.
    """
    from sklearn.metrics import roc_curve

    if metodo not in METODOS_CUTOFF:
        raise ValueError(f"metodo debe ser uno de {METODOS_CUTOFF}, no {metodo!r}")
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype="float64")
    if len(np.unique(y_true)) < 2:
        raise ValueError("El cutoff necesita ambas clases (churn y no churn) en y_true.")

    fpr, tpr, thr = roc_curve(y_true, prob)
    # thr[0] es +inf (nadie en la lista): se excluye de la búsqueda.
    idx = {
        "cruce": 1 + int(np.argmin(np.abs(tpr[1:] - (1 - fpr[1:])))),
        "youden": int(np.argmax(tpr - fpr)),
    }
    elegido = _punto(y_true, prob, fpr, tpr, thr, idx[metodo])
    otro = "youden" if metodo == "cruce" else "cruce"
    ref = _punto(y_true, prob, fpr, tpr, thr, idx[otro])
    ref.pop("matriz_confusion")
    return {
        "metodo": metodo,
        **elegido,
        "tasa_base": float(y_true.mean()),
        "n_evaluado": int(len(y_true)),
        f"referencia_{otro}": ref,
    }


def calcular_cutoff_youden(y_true, prob):
    """Atajo para comparar: el punto de Youden (no es el que se aplica)."""
    return calcular_cutoff(y_true, prob, metodo="youden")


def guardar_cutoff(info, out_dir, model_path, calculado_en, extra=None):
    """Escribe motor2_donante_cutoff.json junto al modelo y devuelve su ruta."""
    artefacto = {
        "metodo": info.get("metodo", METODO_CUTOFF),
        "regla": "alerta = 1 si prob_churn_donante >= cutoff",
        **{k: v for k, v in info.items() if k != "metodo"},
        "calculado_en": calculado_en,
        "modelo_archivo": Path(model_path).name,
        "modelo_sha256": huella_modelo(model_path),
        "generado": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    if extra:
        artefacto.update(extra)
    path = Path(out_dir) / CUTOFF_FILENAME
    with open(path, "w") as f:
        json.dump(artefacto, f, indent=2, ensure_ascii=False)
    return path


def cargar_cutoff(cutoff_path, model_path):
    """
    Lee el cutoff y verifica que corresponde al modelo que se va a usar.

    Si alguien reentrena el modelo y el JSON queda viejo, la huella no coincide
    y se aborta: un umbral calculado para otro modelo daría una lista
    equivocada sin ningún error visible.
    """
    cutoff_path = Path(cutoff_path)
    if not cutoff_path.exists():
        raise SystemExit(
            f"\n[ERROR] Falta el punto de corte de Motor 2 donante: {cutoff_path}\n"
            f"        Se genera al entrenar:  python3 scripts/03_train/02b_train_motor2_donante.py\n"
            f"        o, sin reentrenar, desde las predicciones de test ya existentes:\n"
            f"          python3 scripts/03_train/02b_train_motor2_donante.py --solo-cutoff\n")
    with open(cutoff_path) as f:
        info = json.load(f)
    esperada = info.get("modelo_sha256")
    actual = huella_modelo(model_path)
    if esperada != actual:
        raise SystemExit(
            f"\n[ERROR] El cutoff ({cutoff_path.name}) fue calculado para OTRO modelo.\n"
            f"        sha256 esperado: {esperada}\n"
            f"        sha256 actual:   {actual}\n"
            f"        Vuelve a entrenar (02b) para regenerar modelo y cutoff juntos.\n")
    return info


def aplicar_cutoff(prob, cutoff):
    """Bandera binaria 0/1 (int8): 1 = el donante entra a la lista de retención."""
    return (np.asarray(prob, dtype="float64") >= float(cutoff)).astype("int8")
