#!/usr/bin/env python3
"""
Pruebas sintéticas de la lógica de agregación pledge -> donante (Motor 2).

No necesita datos reales ni artefactos del pipeline: construye a mano los casos
que definen la regla de negocio y verifica que scripts/motor2_donante_common.py
los resuelve como se acordó.

    python3 tests/test_motor2_donante.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from motor2_donante_common import (
    FEATURES_DONANTE, METODO_CUTOFF, adjuntar_contact_id, agregar_a_donante,
    aplicar_cutoff, calcular_cutoff, calcular_cutoff_youden, cargar_cutoff,
    guardar_cutoff,
    preparar_features_donante, resolver_pledge_to_contact,
)

T = pd.Timestamp("2025-06-01")
fallos = []


def check(nombre, cond, detalle=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {nombre}  {detalle}")
    if not cond:
        fallos.append(nombre)


def fila(pledge, contact, label, **kw):
    base = dict(
        pledge_id=pledge, contact_id=contact, snapshot_ts=T,
        label_churn_30d=label, last_payment_amount=100.0, streak_actual=0,
        tenure_days=365, cum_intentos=10, cum_exitos=9, cum_fallos=1,
        n_intentos_30d=1, n_fallos_30d=0, days_since_last_attempt=15.0,
        last_bank_name="BancoA", last_payment_method="PSE",
        success_rate_hist=0.9,
    )
    base.update(kw)
    return base


print("\n=== 1. Regla del label (min del grupo) ===")
df = pd.DataFrame([
    # D1: dos pledges, los DOS cancelan -> donante abandona
    fila("p1", "D1", 1), fila("p2", "D1", 1),
    # D2: dos pledges, solo uno cancela -> donante NO abandona
    fila("p3", "D2", 1), fila("p4", "D2", 0),
    # D3: un solo pledge que cancela -> donante abandona
    fila("p5", "D3", 1),
    # D4: un solo pledge que no cancela
    fila("p6", "D4", 0),
])
ag = agregar_a_donante(df).set_index("contact_id")
check("D1 (ambos cancelan) -> 1", ag.loc["D1", "label_churn_30d"] == 1)
check("D2 (mixto) -> 0", ag.loc["D2", "label_churn_30d"] == 0)
check("D3 (uno solo, cancela) -> 1", ag.loc["D3", "label_churn_30d"] == 1)
check("D4 (uno solo, no cancela) -> 0", ag.loc["D4", "label_churn_30d"] == 0)
check("tasa donante <= tasa pledge",
      ag["label_churn_30d"].mean() <= df["label_churn_30d"].mean(),
      f"({ag['label_churn_30d'].mean():.2f} vs {df['label_churn_30d'].mean():.2f})")

print("\n=== 2. Reglas de agregación ===")
df = pd.DataFrame([
    fila("p1", "D1", 0, last_payment_amount=100.0, streak_actual=0, tenure_days=1000,
         cum_intentos=50, cum_exitos=45, cum_fallos=5, success_rate_hist=0.9,
         days_since_last_attempt=5.0, last_bank_name="BancoA", last_payment_method="PSE",
         n_intentos_30d=2, n_fallos_30d=0),
    fila("p2", "D1", 0, last_payment_amount=40.0, streak_actual=3, tenure_days=30,
         cum_intentos=2, cum_exitos=1, cum_fallos=1, success_rate_hist=0.5,
         days_since_last_attempt=90.0, last_bank_name="BancoB", last_payment_method="TC",
         n_intentos_30d=1, n_fallos_30d=1),
])
a = agregar_a_donante(df).iloc[0]
check("n_pledges_activos = 2", a["n_pledges_activos"] == 2)
check("monto_total_mensual = 140 (sum)", a["monto_total_mensual"] == 140.0)
check("streak_max = 3 (peor racha)", a["streak_max"] == 3)
check("streak_promedio = 1.5", a["streak_promedio"] == 1.5)
check("tenure_max = 1000 / tenure_min = 30",
      a["tenure_max"] == 1000 and a["tenure_min"] == 30)
check("cum_intentos_total = 52 (sum)", a["cum_intentos_total"] == 52)
check("days_since_last_attempt_min = 5 (el pledge VIVO manda)",
      a["days_since_last_attempt_min"] == 5.0)
check("n_bancos_distintos = 2", a["n_bancos_distintos"] == 2)
check("n_metodos_distintos = 2", a["n_metodos_distintos"] == 2)
# ponderado: (0.9*50 + 0.5*2)/52 = 46/52
esperado = (0.9 * 50 + 0.5 * 2) / 52
check("success_rate_hist_pond ponderado por intentos, NO promedio simple",
      abs(a["success_rate_hist_pond"] - esperado) < 1e-6,
      f"({a['success_rate_hist_pond']:.5f} vs esperado {esperado:.5f}; "
      f"promedio simple daría {(0.9+0.5)/2:.5f})")

print("\n=== 3. contact_id ya presente: no debe duplicar columnas (bug corregido) ===")
df = pd.DataFrame([fila("p1", "D1", 0)])
out = adjuntar_contact_id(df, staging_path=None)
check("no crea contact_id_x / contact_id_y",
      "contact_id" in out.columns and "contact_id_x" not in out.columns)
check("no pierde filas", len(out) == 1)

print("\n=== 4. Filas sin contact_id se excluyen ===")
df = pd.DataFrame([fila("p1", "D1", 0), fila("p2", None, 1)])
out = adjuntar_contact_id(df, staging_path=None)
check("descarta la fila sin contact_id", len(out) == 1 and out.iloc[0]["contact_id"] == "D1")

print("\n=== 5. Resolución de contact_id ambiguo -> el más frecuente ===")
stg = pd.DataFrame({
    "pledge_id": ["p1", "p1", "p1", "p2"],
    "contact_id": ["A", "A", "B", "C"],   # p1: A x2, B x1 -> gana A
})
tmp = Path(tempfile.mkdtemp()) / "_stg_test.parquet"
stg.to_parquet(tmp, index=False)
res = resolver_pledge_to_contact(tmp).set_index("pledge_id")
check("p1 -> A (el más frecuente)", res.loc["p1", "contact_id"] == "A")
check("p2 -> C", res.loc["p2", "contact_id"] == "C")

print("\n=== 6. Paridad de imputación train/score (NaN -> 999) ===")
df = pd.DataFrame([fila("p1", "D1", 0, days_since_last_attempt=np.nan)])
ag = agregar_a_donante(df, con_label=False)
check("la agregación deja NaN si no hay actividad previa",
      pd.isna(ag.iloc[0]["days_since_last_attempt_min"]))
prep = preparar_features_donante(ag)
check("preparar_features_donante imputa 999",
      prep.iloc[0]["days_since_last_attempt_min"] == 999)
check("todas las FEATURES_DONANTE presentes y sin NaN",
      prep[FEATURES_DONANTE].notna().all().all())

print("\n=== 7. con_label=False no emite label (snapshot mensual) ===")
ag = agregar_a_donante(pd.DataFrame([fila("p1", "D1", 0)]), con_label=False)
check("sin columna label_churn_30d", "label_churn_30d" not in ag.columns)
check("mismas features que el entrenamiento",
      all(f in ag.columns for f in FEATURES_DONANTE))

print("\n=== 8. Snapshots distintos no se mezclan ===")
df = pd.DataFrame([
    fila("p1", "D1", 1, snapshot_ts=T),
    fila("p2", "D1", 0, snapshot_ts=T + pd.DateOffset(months=1)),
])
ag = agregar_a_donante(df)
check("2 filas (una por snapshot), no 1", len(ag) == 2)
check("cada snapshot conserva su label",
      set(ag["label_churn_30d"]) == {0, 1})

print("\n=== 9. Cutoff: cruce Sens = Espec (aplicado) vs Youden (referencia) ===")
# Caso a mano con 3 escalones de probabilidad (como deja la isotónica) donde
# los dos criterios dan umbrales DISTINTOS:
#   prob 0.50 -> 3 churn, 0 no churn
#   prob 0.20 -> 0 churn, 4 no churn
#   prob 0.01 -> 1 churn, 12 no churn
# Positivos=4, negativos=16.
#   >=0.50: sens 3/4, espec 16/16 -> J 0.75  |Sens-Espec| 0.25   <- Youden
#   >=0.20: sens 3/4, espec 12/16 -> J 0.50  |Sens-Espec| 0.00   <- cruce
prob = np.array([.5] * 3 + [.2] * 4 + [.01] * 13)
y = np.array([1] * 3 + [0] * 4 + [1] + [0] * 12)
check("el método aplicado por defecto es 'cruce'", METODO_CUTOFF == "cruce")
r = calcular_cutoff(y, prob)
check("cruce: elige el escalón 0.20", r["prob_minima_en_alerta"] == 0.2)
check("cruce: sensibilidad = especificidad = 75%",
      abs(r["sensibilidad"] - .75) < 1e-9 and abs(r["especificidad"] - .75) < 1e-9)
check("cruce: cutoff en el punto medio entre escalones (0.105)", abs(r["cutoff"] - 0.105) < 1e-9)
check("cruce: matriz de confusión", r["matriz_confusion"] == {"VP": 3, "FP": 4, "FN": 1, "VN": 12})
check("cruce: registra Youden como referencia (0.35, J 0.75)",
      abs(r["referencia_youden"]["cutoff"] - 0.35) < 1e-9
      and abs(r["referencia_youden"]["youden_j"] - 0.75) < 1e-9)
ry = calcular_cutoff_youden(y, prob)
check("youden: elige el escalón 0.50 (máximo J)", ry["prob_minima_en_alerta"] == 0.5)
check("youden: registra el cruce como referencia",
      abs(ry["referencia_cruce"]["cutoff"] - 0.105) < 1e-9)
check("NO es 0.5 (con clases desbalanceadas 0.5 dejaría fuera a casi todos)",
      r["cutoff"] < 0.5)

al = aplicar_cutoff(prob, r["cutoff"])
check("aplicar_cutoff -> 0/1 int8", al.dtype == np.int8 and set(al) <= {0, 1})
check("7 donantes en alerta", int(al.sum()) == 7)
# Robustez float32: el escalón redondeado a float32 sigue quedando en alerta.
check("float32 del escalón no cambia la decisión",
      int(aplicar_cutoff(prob.astype("float32"), r["cutoff"]).sum()) == 7)

for nombre, f in [("una sola clase -> error", lambda: calcular_cutoff(np.zeros(5), np.linspace(0, 1, 5))),
                  ("método desconocido -> error", lambda: calcular_cutoff(y, prob, metodo="f1"))]:
    try:
        f(); check(nombre, False)
    except ValueError:
        check(nombre, True)

print("\n=== 10. Artefacto JSON amarrado al modelo ===")
d = Path(tempfile.mkdtemp())
modelo = d / "modelo.cbm"
modelo.write_bytes(b"modelo-v1")
p = guardar_cutoff(r, d, modelo, calculado_en="test sintético")
info = cargar_cutoff(p, modelo)
check("se lee el mismo cutoff que se guardó", info["cutoff"] == r["cutoff"])
modelo.write_bytes(b"modelo-v2-reentrenado")
try:
    cargar_cutoff(p, modelo)
    check("cutoff de otro modelo -> aborta", False)
except SystemExit:
    check("cutoff de otro modelo -> aborta", True)
try:
    cargar_cutoff(d / "no_existe.json", modelo)
    check("cutoff ausente -> aborta con instrucción", False)
except SystemExit:
    check("cutoff ausente -> aborta con instrucción", True)

print("\n" + "=" * 60)
if fallos:
    print(f"  {len(fallos)} PRUEBA(S) FALLARON: {fallos}")
    sys.exit(1)
print("  TODAS LAS PRUEBAS PASARON")
