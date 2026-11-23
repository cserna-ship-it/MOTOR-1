# Resultados Motor 2 (churn 30 días) — v2

> ⚠️ **SUPERADO por v3 (2026-07-16).** Estos números corresponden a la config con
> `auto_class_weights="Balanced"`. Se demostró que el balanceo era contraproducente y se
> reemplazó por la config sin pesos (lr=0.03). Resultados vigentes y comparación en
> [comparacion_motor2_v2_alternativas.md](comparacion_motor2_v2_alternativas.md).
> Se conserva este documento como registro del punto de partida.


**Fecha de ejecución:** 2026-07-16
**Script:** `scripts/03_train/02_train_motor2.py`
**Entorno:** `.venv` (CatBoost 1.2.10, scikit-learn 1.9.0, pandas 2.3.3, pyarrow 25.0.0)
**Dataset:** `data/features/motor2_pledge_month.parquet` — 2,548,075 filas, 224,261 pledges

El proceso se ejecutó de principio a fin sin errores y reproduce los números de forma
determinista (`random_seed=42`). A continuación el resumen, la verificación de cada uno
de los 5 cambios de v1→v2, y las salvedades detectadas.

---

## 1. Resumen de métricas (test = snapshots 2025)

| Métrica | Valor | Lectura |
|---|---|---|
| **AUC (calibrado)** | **0.8785** | Buena capacidad de ordenamiento |
| PR-AUC | 0.2812 | ~8× sobre el azar (tasa base 3.5%) |
| Brier crudo → calibrado | 0.1786 → **0.0269** | La calibración es la que hace el score usable |
| ECE test antes → después | 0.3866 → **0.0096** | Reducción de ~40× en error de calibración |
| Recall @ top 20% | **0.8312** | Contactando al 20% con mayor P2 se capturan el 83% de los churners reales |
| Precision @ top 20% | 0.1280 | ~3.7× sobre la tasa base (3.5%) |
| Lift AUC vs B1 (streak) | +0.0564 | Supera la mejor regla trivial |
| Lift AUC vs B2 (tenure) | +0.2095 | Supera holgadamente el lookup por antigüedad |

**Split:** Train 1,173,916 filas (2024) · Test 1,374,159 (2025). Tasa churn: train 4.43% / test 3.48%.
**Overlap de pledges train↔test:** 97,359 (56.6% del test) — esperado y documentado, no es leakage (mismas personas, snapshots distintos).

**Veredicto:** los resultados tienen sentido. El modelo supera claramente a los tres
baselines triviales, la calibración funciona como se esperaba, y la métrica operacional
(recall 83% en el top 20%) es la clase de resultado accionable para una campaña de retención.

---

## 2. Verificación de los 5 cambios de v2

| # | Cambio | Evidencia en esta corrida | Estado |
|---|---|---|---|
| 1 | Eliminar `sample_frac=0.5` | `n_train_used = 1,173,916` = 100% del train (`config_v2.sample_frac = 1.0`) | ✅ |
| 2 | `auto_class_weights="Balanced"` | Probabilidades crudas concentradas en ~0.42 (media 0.4214), típico del balanceo | ✅ |
| 3 | Calibración isotónica | ECE test 0.387 → 0.0096; Brier 0.179 → 0.027; `motor2_calibrador.pkl` generado; parquet con `prob_churn` + `prob_churn_cruda` | ✅ |
| 4 | `validate_temporal_integrity` | `[OK anti-leakage]` en train (149,622 pledges) y test (171,998): days≥0, conteos monótonos, tenure≥0 | ✅ |
| 5 | Re-orden canónico pre-split | Ejecución determinista; val holdout = últimos 10% cronológicos | ✅ |

**Efecto de la calibración** (mapeo crudo → calibrado del `.pkl`): scores crudos < 0.4 → ~1%,
0.6 → 24%, ≥0.7 → 92%. Es decir, el calibrador colapsa la masa sobreestimada del balanceo
hacia la tasa real y solo asigna probabilidad alta a los scores crudos extremos. La media
calibrada en test es 4.44% frente a la tasa real 3.48%.

---

## 3. Salvedades / puntos a vigilar

Ninguna invalida los resultados, pero conviene dejarlas registradas:

1. **Early stopping muy temprano — el modelo usa solo 4 árboles.**
   El detector de overfitting corta en `bestIteration = 3` (de 150 posibles): la AUC en el
   val interno (últimos 10% de 2024) alcanza su máximo, 0.826, en el árbol 3 y empeora
   después. El modelo final es, por tanto, muy superficial. No es un error —aún así logra
   AUC 0.878 en test y supera a todos los baselines— pero implica que:
   - el tope `iterations=150` nunca es relevante;
   - la señal está muy concentrada en pocas features (ver punto 3);
   - vale la pena, si se quiere exprimir más, probar `learning_rate` más bajo (p.ej. 0.03)
     con más iteraciones, o revisar el tamaño/representatividad del val interno.

2. **`ece_val_despues ≈ 0` (1e-17) es in-sample.**
   El calibrador se ajusta sobre el mismo val set donde se mide ese ECE, así que el ~0 es
   esperado y no debe leerse como calibración perfecta. La métrica honesta es el **ECE en
   test = 0.0096**, que de todos modos es excelente.

3. **Varias features con importancia 0.**
   `n_intentos_30d`, `n_fallos_30d`, `cum_intentos`, `days_since_last_attempt` y las tres
   categóricas (`last_payment_method`, `last_bank_name`, `last_gateway`) quedan en 0. Es en
   buena parte consecuencia del modelo de 4 árboles: nunca necesitó esas variables. El peso
   recae en `tenure_days` (35.9%), `streak_actual` (27.0%), `cum_exitos` (16.0%),
   `last_attempt_success` (12.2%) y `success_rate_hist` (5.6%) — todas coherentes con el
   negocio (antigüedad y racha de pagos como principales predictores de churn).

4. **Ligera sobreestimación de la probabilidad media (4.4% calibrada vs 3.5% real en test).**
   El calibrador se entrena sobre 2024 (tasa 4.4%) y se aplica a 2025 (tasa 3.5%), por lo que
   queda anclado a la tasa de 2024. Es un efecto de drift interanual, no un defecto del
   método; el ordenamiento (AUC) y el ECE no se ven afectados de forma material.

---

## 4. Artefactos generados

| Archivo | Contenido |
|---|---|
| `results/motor2/motor2_catboost.cbm` | Modelo CatBoost serializado |
| `results/motor2/motor2_calibrador.pkl` | Calibrador isotónico |
| `results/motor2/resultados_motor2.json` | Métricas completas (incluye bloque `calibracion`) |
| `results/motor2/feature_importance_motor2.csv` | Importancia de features |
| `results/motor2/test_predictions_motor2.parquet` | 1,374,159 predicciones con `prob_churn` (calibrada, usar esta) y `prob_churn_cruda` |

---

## 5. Conclusión

Los cambios de v2 están correctamente implementados y producen un modelo coherente y
operacionalmente útil: **AUC 0.878**, calibración casi perfecta en test (**ECE 0.0096**) y
un **recall del 83% en el top 20%** de donantes priorizados. La única observación de fondo es
el corte de early stopping en 4 árboles, que deja margen para explorar hiperparámetros si se
busca más desempeño, pero no compromete la validez de los resultados actuales.
