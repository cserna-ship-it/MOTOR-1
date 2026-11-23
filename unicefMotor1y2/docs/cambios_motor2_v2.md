# Cambios Motor 2 — v1 → v2

Script afectado: `scripts/03_train/02_train_motor2.py`

---

## 1. Eliminación de `sample_frac=0.5`

**Qué cambió:** Se reemplazó `train.sample(frac=0.5)` por usar `train` directamente. Desapareció la variable intermedia `train_s`.

**Por qué:** No hay restricciones de RAM que justifiquen descartar la mitad de los datos de entrenamiento. Submuestrear reduce artificialmente la información disponible para el modelo, lo que baja el AUC. No hay ningún beneficio técnico en hacerlo cuando la memoria lo permite.

---

## 2. `auto_class_weights="Balanced"`

**Qué cambió:** Se agregó `auto_class_weights="Balanced"` al `CatBoostClassifier`.

**Por qué:** El target (`label_churn_30d`) está fuertemente desbalanceado: aproximadamente el 4-5% de las observaciones son churners. Sin balanceo, CatBoost puede alcanzar 95%+ de accuracy simplemente prediciendo "no churn" en todos los casos, sin detectar a ningún churner real. `Balanced` pondera cada clase inversamente a su frecuencia, dándole más peso a los churners (la clase accionable para campañas de retención). El costo es que las probabilidades crudas quedan sobreestimadas hacia la clase positiva, lo cual se corrige con el calibrador (punto 3).

---

## 3. Calibración isotónica (`IsotonicRegression`)

**Qué cambió:** Tras el entrenamiento se ajusta un `IsotonicRegression` sobre el validation set interno (el último 10% del train). El calibrador se guarda en `results/motor2/motor2_calibrador.pkl`. Las predicciones de test en el parquet ahora tienen dos columnas: `prob_churn` (calibrada, la que se debe usar en producción) y `prob_churn_cruda` (solo para referencia).

**Por qué:** `auto_class_weights="Balanced"` introduce un sesgo sistemático: el modelo sobreestima P(churn) porque internamente trata ambas clases como equiprobables cuando en realidad la tasa real es ~5%. Sin calibración, usar el score directamente como probabilidad daría valores incorrectos (p.ej. predecir 40% de probabilidad de churn cuando la tasa real es 5%). El calibrador aprende a mapear el score crudo a probabilidades reales. Se usa `IsotonicRegression` (no Platt/sigmoid) porque es no paramétrica y puede corregir cualquier forma de sesgo monótono, sin asumir que el error es simétrico.

La mejora se reporta mediante ECE (Expected Calibration Error) antes y después, tanto en val como en test.

---

## 4. Validación de integridad temporal (`validate_temporal_integrity`)

**Qué cambió:** Se agregó la función `validate_temporal_integrity(df, nombre)` y se llama sobre `train` y `test` antes de entrenar. Si alguna validación falla, el script aborta con `AssertionError`.

**Por qué:** El anti-leakage de Motor 2 se garantiza aguas arriba en `01_motor2_dataset.py` mediante `merge_asof(direction="backward", allow_exact_matches=False)`. Sin embargo, un error silencioso en el pipeline (un reorden accidental, un cambio en los tipos de fecha, una actualización del parquet) podría romper ese invariante sin que el entrenamiento falle. La validación actúa como red de seguridad explícita dentro del script de entrenamiento.

Las tres invariantes que se verifican son:

- `days_since_last_attempt >= 0` para todos los valores reales (excluyendo el sentinel 999 que indica "sin intentos previos"). Un valor negativo significaría que el merge_asof tomó un intento posterior al snapshot.
- `cum_intentos`, `cum_exitos`, `cum_fallos` son monótonos no decrecientes dentro de cada pledge ordenado por `snapshot_ts`. Un decremento indicaría que un snapshot más reciente "ve" menos historial que uno anterior, lo que rompería el anti-leakage.
- `tenure_days >= 0`. Un valor negativo significaría que el snapshot se generó antes de que el pledge existiera, lo que indica que el filtro de snapshots inválidos en `01_motor2_dataset.py` no funcionó.

---

## 5. Re-orden canónico antes del split

**Qué cambió:** Se agregó `df = df.sort_values(["pledge_id", "snapshot_ts"]).reset_index(drop=True)` después de cargar el parquet.

**Por qué:** El parquet puede no preservar el orden de filas entre escrituras (depende de la versión de PyArrow y del sistema de archivos). El holdout de validación interna (`tr_idx` / `val_idx`) se construye tomando los últimos N índices del dataframe, asumiendo orden cronológico. Sin re-ordenar explícitamente, el val set sería arbitrario y no reproducible entre re-ejecuciones. Con el re-orden, el último 10% del train corresponde a los snapshots más recientes de 2024, lo que además lo hace más parecido al test real (2025).

---

## Archivos generados (nuevos respecto a v1)

| Archivo | Descripción |
|---|---|
| `results/motor2/motor2_calibrador.pkl` | Calibrador isotónico serializado |
| `results/motor2/test_predictions_motor2.parquet` | Ahora incluye columna `prob_churn_cruda` además de `prob_churn` |
| `results/motor2/resultados_motor2.json` | Incluye bloque `"calibracion"` con ECE val/test antes y después |
