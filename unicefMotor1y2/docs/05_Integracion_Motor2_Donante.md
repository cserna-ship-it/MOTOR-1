# Integración del Motor 2 a nivel donante — bitácora de la integración

Este documento registra **cómo se integró en `v9_entregable/` el paquete de
cambios entregado por el equipo matemático** (carpeta `Cambiosenelmotor2`,
bitácora original en `docs/Cambios_Motor_2.pdf`), qué se modificó respecto a lo
entregado y por qué.

La bitácora original explica el *porqué* del cambio (regla de negocio, diseño de
las features). Este documento cubre solo el *cómo quedó* en el entregable.

---

## 1. Dónde quedó cada archivo

| Archivo entregado | Ubicación final | Nota |
|---|---|---|
| `01_motor2_dataset_MODIFICADO.py` | `scripts/02_features_motor2/01_motor2_dataset.py` | reemplaza al de producción (18 → 19 columnas) |
| `01b_motor2_dataset_donante.py` | `scripts/02_features_motor2/01b_motor2_dataset_donante.py` | reescrito sobre el módulo común |
| `02_train_motor2_donante.py` | `scripts/03_train/02b_train_motor2_donante.py` | renombrado a `02b_` para no chocar con `02_train_motor2.py` |
| `04_score_motor2_donante.py` | `scripts/03_train/04b_score_motor2_donante.py` | renombrado a `04b_` por lo mismo |
| `00_diagnostico_donante.py` | `docs/00_diagnostico_donante.py` | sin cambios; es read-only |
| `Cambios_Motor_2.pdf` | `docs/Cambios_Motor_2.pdf` | sin cambios |
| — | `scripts/motor2_donante_common.py` | **nuevo**: módulo compartido |

**Por qué no quedaron en `docs/` como decía la sección 6 de la bitácora.** Los
scripts `01b`, `02` y `04` resuelven sus rutas con
`ROOT = Path(__file__).resolve().parents[2]`. Desde `docs/` eso apunta al
directorio *padre* del entregable, no al entregable — todas las rutas de datos
quedarían mal. La profundidad que el código asume es la de `scripts/<subcarpeta>/`,
que además es donde vive el resto del pipeline. `00_diagnostico_donante.py` usa
`parents[1]` y sí corresponde a `docs/`, así que ese sí se quedó ahí.

---

## 2. Correcciones aplicadas sobre lo entregado

### 2.1 `contact_id` duplicado — rompía la ejecución (bloqueante)

`01_motor2_dataset.py` modificado escribe ahora `contact_id` en
`motor2_pledge_month.parquet`. Pero `01b` volvía a resolverlo desde el staging y
lo pegaba con un merge:

```python
pledge_df = pd.read_parquet(MOTOR2_PLEDGE)      # ya trae contact_id
df = pledge_df.merge(pledge_to_contact, on="pledge_id", how="left")
df["contact_id"]                                 # KeyError
```

Al chocar dos columnas con el mismo nombre, pandas las renombra a
`contact_id_x` / `contact_id_y` y la línea siguiente falla. Los dos cambios
funcionaban por separado, pero no juntos — que es como hay que aplicarlos.

**Corregido** en `adjuntar_contact_id()`: si la columna ya viene en el parquet se
usa tal cual; el camino de releer el staging queda solo como respaldo para
parquets generados con la versión anterior del script.

### 2.2 Imputación divergente entre entrenamiento y puntuación (sesgo silencioso)

`02_train_motor2_donante.py` imputaba antes de entrenar:

```python
df["days_since_last_attempt_min"] = df[...].fillna(999)
df["monto_total_mensual"]         = df[...].fillna(0)
```

`04_score_motor2_donante.py` **no** hacía ninguna imputación. Un donante recién
activado (sin intentos previos al snapshot) llegaba al modelo con `NaN` donde el
entrenamiento vio `999`. CatBoost admite `NaN` en features numéricas, pero lo
trata como un valor propio con su propia rama: esas filas se habrían puntuado
por un camino que el modelo nunca vio en entrenamiento, sin lanzar ningún error.

**Corregido**: la imputación vive en `preparar_features_donante()` y la llaman
los dos scripts.

### 2.3 Duplicación de `agregar_a_donante()`

La propia bitácora la señalaba como deuda a saldar antes de producción, y las dos
correcciones anteriores son exactamente el tipo de cambio que se aplica en un
lado y se olvida en el otro.

**Corregido**: `scripts/motor2_donante_common.py` es la única fuente de verdad de
`FEATURES_DONANTE`, `adjuntar_contact_id()`, `agregar_a_donante()` y
`preparar_features_donante()`. Los tres scripts lo importan.

---

## 3. Guardarraíles que se agregaron al entrenamiento de donante

`02_train_motor2.py` (pledge-level) tiene protecciones que el borrador de donante
no traía. Se portaron para que los dos modelos tengan el mismo contrato:

- **Coherencia dataset ↔ corte**: aborta si el último snapshot del parquet no es
  el mes de la fecha de corte. Sin esto, `run_all.sh` (que salta las etapas cuya
  salida ya existe) reentrenaría en silencio sobre features de un corte anterior.
- **Censura del label**: avisa cuántas filas de test tienen la ventana de 30 días
  todavía abierta al corte.
- **Modo scoring**: si el test tiene una sola clase, entrena y escribe
  predicciones sin intentar calcular AUC (que fallaría).
- **`resultados_motor2_donante.json`** con la misma estructura que el
  pledge-level, más `advertencia_comparabilidad`.
- **`test_predictions_motor2_donante.parquet` + `.csv`**, para que la salida sea
  consumible igual que la del pledge-level.
- **Baseline B3 (`n_pledges_activos`)**: es la señal *nueva* que aporta el cambio
  a donante. Si el modelo no le gana claramente a un simple lookup sobre ella, la
  agregación no está aportando nada y hay que revisarlo.

---

## 4. Cambios en los runners

**`run_all.sh`**

- `verify_step "motor2_pledge_month"`: 18 → **19** columnas (lo advertía la
  bitácora original).
- Etapa nueva `M2d`: genera el dataset de donante y entrena el modelo, después
  del entrenamiento pledge-level. Es idempotente como el resto (se salta si su
  salida existe).

**`run_monthly.sh`**

- Etapa nueva `E2`: puntúa el snapshot del mes a nivel donante. Se **salta con
  `SKIP`, sin fallar**, si `results/motor2_donante/motor2_donante_catboost.cbm` no
  existe — es una salida adicional, no un pre-requisito de la tabla maestra.

---

## 5. Qué queda pendiente

1. **Enriquecimiento demográfico.** El borrador tenía un bloque que solo imprimía
   `[PENDIENTE]`; se eliminó porque no hacía nada. Si se quiere, la vía correcta
   es la que ya anotaba el autor: usar `motor1_features.parquet` como lookup de
   las columnas normalizadas (`gender_norm`, `age_band`, `ses_level`,
   `state_norm`) en vez de reprocesar `contact_unified.parquet`.

2. **El modelo de donante no alimenta la tabla maestra.** `Tabla_Donantes_P1_P2`
   sigue tomando P2 del modelo pledge-level. Decidir si P2 pasa a ser la
   probabilidad de donante, o si conviven las dos columnas, es una decisión de
   negocio pendiente.

3. **Umbral de intervención.** Lo pide el cliente y no es parte de estos scripts:
   se decide con el modelo ya entrenado, mirando el trade-off
   precision/recall en `resultados_motor2_donante.json`.

4. **Sin features categóricas.** El modelo de donante entrena con
   `cat_features=[]`. Vale la pena medir si agregar el banco/método *modal* del
   donante (no el de un pledge suelto) recupera señal.

---

## 6. Resultados de la primera corrida con datos reales (corte 2025-12-31)

`run_all.sh` completo desde los CSV crudos: **20 PASS, 0 FAIL**. Los 6 parquets de
preprocesamiento coinciden exactamente con los conteos esperados, y
`motor2_pledge_month` verifica 2.548.075 × **19** columnas.

### 6.1 El hallazgo principal: en estos datos el cambio casi no mueve nada

| | valor |
|---|---|
| Donantes con >1 pledge (histórico) | 7.972 de 217.230 — **3,67%** |
| Donante-snapshots con >1 pledge **activo a la vez** | 8.882 de 2.515.029 — **0,35%** |
| Pledges activos por donante-snapshot | media **1,004** (máx. 6) |
| Filas del dataset | 2.524.247 → 2.515.029 (**−0,4%**) |
| Tasa de churn | pledge 0,0365 → donante 0,0362 |
| Filas donde el label de donante **difiere** del pledge-level | **941** |

Esas 941 filas son el universo entero donde el modelo pledge-level estaba
respondiendo la pregunta equivocada. Es real y la corrección es la correcta,
pero es **0,04% del dataset**.

### 6.2 Métricas (test 2025) — prácticamente empatadas

> **Nota:** esta sección documenta la corrida ORIGINAL de la integración, ANTES
> de la corrección de calibración descrita en `docs/07_Informe_calibracion.md`.
> Se deja tal cual como registro histórico. Las cifras vigentes son las de la
> columna "después" de ese informe y las del `README.md`.

| | pledge | donante | *(vigente hoy: pledge / donante)* |
|---|---|---|---|
| AUC | 0,8963 | 0,8932 | *0,9028 / 0,8984* |
| PR-AUC | 0,3672 | 0,3698 | *0,4088 / 0,3769* |
| Recall @ top 20% | 0,8552 | 0,8410 | *0,8525 / 0,8398* |
| Precision @ top 20% | 0,1237 | 0,1298 | *0,1277 / 0,1306* |
| Brier calibrado | 0,0267 | 0,0266 | *0,0241 / 0,0240* |

La lista de retención del top 20% coincide en **85,6%** entre los dos modelos.

**`B3_lookup_n_pledges` da AUC = 0,5006**: `n_pledges_activos` — la feature
nueva, la que motivaba todo el cambio — no tiene por sí sola **ningún** poder
predictivo en estos datos. Consistente con que el 99,65% de los donantes tiene
un solo pledge activo.

En el subconjunto multi-pledge (5.444 filas de test) el modelo baja a AUC 0,7291
y PR-AUC 0,0914: justo donde el cambio debería aportar, hay muy pocos casos para
aprender el patrón.

### 6.3 Dos observaciones de calidad (preexistentes, no del cambio)

> **Las dos se corrigieron después.** Se dejan documentadas porque explican por
> qué el modelo de donante paraba en 20 árboles. Ver `docs/07_Informe_calibracion.md`.

**a) `boost_from_average=False` en los dos modelos.** Arrancan en p=0,5 y el
early stopping sobre AUC los corta antes de que las probabilidades crudas bajen
a la tasa base. El pledge-level alcanza 60 árboles (media cruda 0,052 vs real
0,035); el de donante para en 21 (media cruda **0,139** vs real 0,033). De ahí
que su ECE crudo sea 0,1086 contra 0,0213. **El calibrador isotónico lo corrige**
(→ 0,0125) y el ranking no se ve afectado, así que no invalida nada — pero
activar `boost_from_average=True` daría probabilidades crudas sanas y
probablemente más árboles antes del corte.

**b) El validation set de early stopping no es representativo.** Se toma como el
último 10% del train ordenado por `contact_id` / `pledge_id`. Los IDs de
Salesforce son aproximadamente cronológicos, así que ese orden correlaciona con
la antigüedad del donante: la tasa de churn por decil va de **0,0174 a 0,0623**
(3,6x). El val queda con 1,5x la tasa de churn del train. Afecta a los dos
modelos por igual. Convendría usar un corte temporal explícito (p. ej. los
últimos 2 meses del train) en vez de un corte posicional.

### 6.4 Verificación de la ruta mensual

Se punteó el snapshot 2025-12 con `04b_score_motor2_donante.py` y se comparó
contra las predicciones que escribió el entrenamiento para esas mismas filas:

```
donantes cruzados: 126.509
diferencia máxima de probabilidad: 0.000e+00
```

Paridad exacta — la agregación e imputación compartidas hacen lo que deben. (Con
el código original esta comparación no se podía ni correr: `01b` fallaba con
`KeyError`, y de haber pasado, la imputación divergente habría dado
probabilidades distintas para los donantes sin actividad previa.)

### 6.5 Recomendación

El cambio está **bien hecho y correctamente integrado**, y conceptualmente es la
forma correcta de plantear el problema. Pero con estos datos **no cambia las
decisiones operativas**: mismo AUC, 85,6% de solape en la lista de retención,
y la feature que lo motivaba no discrimina.

Vale la pena mantenerlo (el costo es 12 segundos de pipeline y el problema queda
bien planteado para cuando UNICEF tenga más donantes multi-pledge), pero **no
justifica reemplazar el modelo pledge-level en la tabla maestra**. Antes de
tomar esa decisión conviene confirmar con negocio si el 3,67% de donantes
multi-pledge refleja la realidad o si hay pledges que deberían estar
consolidados bajo un mismo `contact_id` y no lo están.
