# Entregable UNICEF Colombia — v8 

**Iridian · Julio 2026**

Paquete de entrega del pipeline predictivo. Esta versión reorganiza el proyecto
en una **estructura plana** (sin las subcarpetas por versión de entregas
anteriores) e incorpora los ajustes acordados con el cliente sobre el Motor 1 y el
Motor 2 (ver sus respectivas secciones abajo).

## Qué incluye este paquete

Se entrega el **código y los insumos**, no los artefactos generados:

| Incluido | Se genera al correr el pipeline |
|---|---|
| `scripts/`, `docs/`, `diccionario/`, `README.md`, `requirements.txt` | `data/unified/`, `data/staging/`, `data/features/` |
| `raw/` — los 9 CSV exportados del CRM | `data/entregable/` (tabla maestra) |
| | `results/` (modelos, calibradores, métricas, predicciones) |

Todo lo de la columna derecha es reproducible con `bash scripts/run_all.sh`. El
`.gitignore` lo excluye para que el paquete no crezca con artefactos.

**Escenario de referencia de esta entrega:** el modelo se entrena con corte
`2025-12-31` (último año cerrado, para poder medirlo contra 12 meses con desenlace
conocido) y la operación se simula al **3 de enero de 2026**: la lista de
cancelaciones corresponde al snapshot `2026-01-01` y las fechas de cobro se
proyectan desde el 3 de enero.

## Estructura

```
v8_entregable/
├── README.md
├── requirements.txt
├── raw/                      ← CSVs crudos del cliente (9 archivos)
├── docs/                     ← documentación (preprocesamiento, features, mapeo de columnas)
├── diccionario/              ← Diccionario_Variables_UNICEF_Colombia.xlsx
├── scripts/
│   ├── run_all.sh            ← ejecuta y valida el pipeline completo
│   ├── ORDEN_DE_EJECUCION.md ← secuencia + insumos/salidas por script
│   ├── 00_preprocesamiento/  ← 00_unify_raw_data, 01_create_staging
│   ├── 01_features_motor1/   ← bank_normalization, enrich_with_contact,
│   │                            build_allocation_features, build_incremental_features (YTD mensual)
│   ├── 02_features_motor2/   ← 01_motor2_dataset, 01b_motor2_dataset_donante
│   ├── 03_train/             ← 01_train_motor1, 02_train_motor2, 02b_train_motor2_donante,
│   │                            03_timing_reintento_motor1, 04_score_motor2,
│   │                            04b_score_motor2_donante
│   ├── 04_features_motor3/   ← 01_tabla_donantes_p1_p2
│   ├── config.py             ← resuelve la fecha de corte (CLI > env > default)
│   ├── motor2_donante_common.py ← reglas de agregación pledge→donante (compartidas
│   │                            por el dataset, el entrenamiento y la puntuación)
│   ├── run_all.sh            ← re-entrenamiento completo
│   └── run_monthly.sh        ← inferencia mensual de Motor 1
├── data/
│   ├── unified/              ← 5 entidades unificadas (salida de 00_unify)
│   ├── staging/              ← stg_payment_attempts (tabla canónica de intentos)
│   ├── features/             ← motor1_features, allocation_features, motor2_pledge_month,
│   │                            motor2_donante_month, bank_canonical,
│   │                            motor2_snapshot_<AAAA_MM_01> (puntuación mensual)
│   └── entregable/           ← Tabla_Donantes_P1_P2_<CUTOFF> (.parquet y .csv)
└── results/
    ├── motor1/               ← modelo .cbm, calibrador .pkl, métricas, predicciones, timing de reintento
    ├── motor2/               ← modelo .cbm, calibrador .pkl, métricas, feature importance,
    │                            predicciones de test, scoring_motor2_<AAAA_MM_DD> (mensual)
    └── motor2_donante/       ← lo mismo, pero con el donante como unidad de predicción
```

### Motor 2: dos unidades de predicción

Desde esta versión conviven **dos** modelos de churn, entrenados con la misma
config y sobre las mismas features de base:

| Carpeta | Unidad | Pregunta que responde |
|---|---|---|
| `results/motor2/` | `(pledge, snapshot)` | ¿este compromiso concreto se cancela este mes? |
| `results/motor2_donante/` | `(donante, snapshot)` | ¿esta persona deja de donar del todo este mes? |

Un donante con dos pledges que cancela uno y mantiene el otro **no** abandonó el
programa — sigue donando. El modelo pledge-level no distingue ese caso; el de
donante sí. **Ninguno reemplaza al otro:** el pledge dice *qué cobro arreglar*, el
donante dice *a quién llamar*.

📄 **Todo el Motor 2 en un solo documento: [`README_MOTOR2.md`](README_MOTOR2.md)**  
## Ajustes de esta versión (Motor 1)

| # | Ajuste | Archivo | Línea(s) |
|---|--------|---------|----------|
| 1 | **Entrenamiento con el 100% del train** (`sample_frac = 1.0`, 2022-2024) | `scripts/03_train/01_train_motor1.py` | 306 |
| 2 | **Orden por pledge dentro del script de entrenamiento**: re-ordena por `(pledge_id, attempt_ts, payment_id)` tras cargar cada slice (deshace el barajado de `df.sample()`) | `scripts/03_train/01_train_motor1.py` | 123 (clave), 218 (re-orden) |
| 3 | **Balanceo automático de clases** (`auto_class_weights = "Balanced"`) | `scripts/03_train/01_train_motor1.py` | 342 |
| 4 | **Validación anti-leakage en el propio script**: comprueba días≥0, conteos monótonos y que el `shift(1)` upstream esté intacto; aborta si falla | `scripts/03_train/01_train_motor1.py` | 244 (función), 312 y 370 (llamadas) |
| 5 | **Estructura plana**: sin subcarpetas por versión; todo cuelga de `v8_entregable/` (`raw/`, `data/`, `results/`, `scripts/`) | toda la carpeta | — |
| 6 | **Fix de días negativos**: `tenure_days_at_2025_12_31` anclado a la primera evidencia real de actividad (ver nota abajo) | `scripts/04_features_motor3/01_tabla_donantes_p1_p2.py` | 258-286 |
| 7 | **Método de pago**: el modelo usa `payment_method` en lugar de `payment_submethod` (frecuentemente vacío); se propaga a `motor1_features` | modelo: `scripts/03_train/01_train_motor1.py` (143); propagación: `scripts/01_features_motor1/02_enrich_with_contact.py` (168, 177) | ver celda |


### ¿Por qué había días negativos y cómo se corrigió?

La columna de tenure —`tenure_days_at_<CUTOFF_SLUG>`, p. ej.
`tenure_days_at_2025_12_31`— medía `corte − pledge_start_date`. En ~1.45%
de los pledges, `npsp__Start_Date__c` es una fecha **futura** (2026): no es el
inicio histórico del donante sino la fecha del ciclo/renovación vigente. Eso
producía tenures negativos aunque el donante llevaba **todo el año pagando**
(esos pledges tenían intentos reales en 2025).

La corrección ancla el tenure a la **primera evidencia real de actividad**:
`min(pledge_start_date, primer intento observado en toda la historia)`. En el
caso normal domina `pledge_start_date` (sin cambios); en el caso roto domina el
primer intento real. Como los datos terminan en 2025, el resultado queda siempre
≥ 0 sin inventar valores ni descartar donantes.

## Ajustes del Motor 2 (churn 30 días) — v3

> **Lea esto primero si le confunde "pledge" vs. "donante".** El Motor 2 tiene
> **dos modelos que conviven** (ver "Motor 2: dos unidades de predicción" arriba).
> El modelo a nivel donante **no reemplazó** al de pledge: se agregó al lado.
>
> Esta sección describe la **base compartida** (cómo se construyen las features,
> el split temporal, el target) y el **modelo pledge-level**, que es el que
> alimenta la tabla maestra. Lo específico del modelo de donante está en
> "Motor 2 a nivel donante" más abajo y, con todo el detalle, en
> `README_MOTOR2.md` y `docs/05_Integracion_Motor2_Donante.md`.

El Motor 2 pledge-level predice, para cada `(pledge, snapshot mensual)`, la
probabilidad de que ese compromiso se cancele en los 30 días siguientes
(`label_churn_30d`). El dataset se construye en
`scripts/02_features_motor2/01_motor2_dataset.py` y el modelo en
`scripts/03_train/02_train_motor2.py`.

El modelo a nivel donante parte del **mismo** dataset, lo agrega a
`(contact_id, snapshot)` y redefine el label: un donante abandona solo si
cancela **todos** sus compromisos activos dentro de la misma ventana de 30 días.

**Split temporal**, derivado de la fecha de corte: `test = snapshots del año del
corte`, `train = snapshots de años anteriores`. Con el corte por defecto
(`2025-12-31`): train = snapshots 2024, test = snapshots 2025.

**Target:** `label_churn_30d` = 1 si `npsp__EndDate__c` cae en
`(snapshot, snapshot + 30d]`. Depende solo de la fecha de cancelación contractual,
no de `payment_status`: un pledge puede fallar muchos cobros sin churnear si el
donante no cancela formalmente.

| # | Ajuste | Detalle |
|---|--------|---------|
| 1 | **Entrenamiento con el 100% del train** (`sample_frac = 1.0`) | Se eliminó el submuestreo `frac=0.5` de v1; no había límite de RAM que lo justificara. |
| 2 | **SIN balanceo de clases** (`auto_class_weights` **NO** se usa) | A diferencia del Motor 1, el balanceo resultó **contraproducente** aquí (ver abajo). Se entrena con Logloss estándar. |
| 3 | **`learning_rate = 0.03`, `iterations = 800`** | Convergencia más lenta y robusta; el early stopping (paciencia 15) decide el corte real (~59 árboles). |
| 4 | **Calibración isotónica** (`IsotonicRegression`) | Se ajusta sobre el validation set y se guarda en `results/motor2/motor2_calibrador.pkl`. Produce la columna `prob_churn` (calibrada, la operativa) y `prob_churn_cruda` (referencia). |
| 5 | **Validación anti-leakage en el propio script** (`validate_temporal_integrity`) | Comprueba `days_since_last_attempt ≥ 0`, conteos acumulados monótonos por pledge y `tenure_days ≥ 0`. Aborta si falla. |
| 6 | **Re-orden canónico** por `(pledge_id, snapshot_ts)` | Garantiza un holdout de validación determinista y reproducible entre ejecuciones. |
| 7 | **Predicciones también en CSV** | Además del `.parquet` (insumo del Motor 3), se exporta `test_predictions_motor2.csv` — mismo contenido, ordenado por `prob_churn` descendente, legible en Excel. |
| 8 | **Fecha de corte configurable en el entrenamiento** | El split ya no está hard-codeado: `test = año del corte`, `train = años anteriores`. Con el default (`2025-12-31`) el resultado es idéntico al de v8. |
| 9 | **Guard de coherencia dataset ↔ corte** | Antes de entrenar compara el último snapshot del dataset contra el mes de corte y aborta con instrucciones si no coinciden. Evita el error silencioso de reentrenar sobre las features de un corte anterior (`run_all.sh` salta las etapas cuya salida ya existe). |
| 10 | **Detección de censura del label** | Un label es observable solo si `snapshot + 30d <= corte`. Los snapshots del mes de corte tienen la ventana abierta, así que como churn solo aparecen las cancelaciones ya registradas en el CRM. El script reporta cuántas filas de test están censuradas (`corte.n_test_label_censurado` en `resultados_motor2.json`) y avisa que esas métricas son optimistas. |
| 11 | **Modo scoring** | Si el test queda con una sola clase (corte en el mes en curso), entrena, calibra y escribe predicciones sin reportar métricas de test, en vez de fallar en `roc_auc_score`. |
| 12 | **Puntuación mensual separada del entrenamiento** | `scripts/03_train/04_score_motor2.py` aplica el modelo guardado al snapshot del mes en curso. Ver "Entrenar vs. puntuar" abajo. |

### ¿Por qué el Motor 2 NO usa `Balanced` pero el Motor 1 sí?

El target de churn está desbalanceado (~4-5% de churners, ratio 21.6:1). El
instinto sería activar `auto_class_weights = "Balanced"` como en el Motor 1, pero
se probó empíricamente (`docs/02b_experimento_motor2.py`) y **empeoraba
los resultados**:

- Con `Balanced` el early stopping cortaba en **3-4 árboles** (la métrica de
  validación saturaba de inmediato). Sin pesos, el modelo entrena ~59 árboles con
  normalidad.
- Quitar `Balanced` subió **AUC 0.878 → 0.896** y **PR-AUC 0.281 → 0.367 (+30%)**.
  El calibrador no puede recuperar esto: la calibración es monótona, no cambia el
  ordenamiento de los casos.
- `Balanced` distorsiona las probabilidades crudas (las empuja a ~0.42 cuando la
  tasa real es ~4%); el calibrador luego tiene que deshacer ese sesgo. Sin
  `Balanced`, las probabilidades crudas ya salen casi calibradas
  (ECE crudo 0.39 → 0.010) y el calibrador solo hace un ajuste fino.

**En resumen:** el balanceo ayuda al Motor 1 pero perjudica al Motor 2. El
calibrador se mantiene porque es barato, entrega la columna `prob_churn` lista para
producción y protege ante drift interanual. El análisis completo está en
`docs/comparacion_motor2_v2_alternativas.md`.

### Resultado final del Motor 2 (test 2025)

Cifras vigentes, después de la corrección de calibración de septiembre de 2026
(ver `docs/07_Informe_calibracion.md`). Entre paréntesis, el valor anterior.

| Métrica | pledge | donante |
|---|---|---|
| AUC (calibrado) | **0.9028** (era 0.896) | **0.8984** (era 0.893) |
| PR-AUC | **0.4088** (era 0.367) | **0.3769** (era 0.370) |
| ECE test (post-calibración) | **0.0023** (era 0.0085) | **0.0053** (era 0.0125) |
| Brier calibrado | 0.0241 | 0.0240 |
| Recall @ top 20% | 0.853 | 0.840 |
| Precision @ top 20% | 0.128 | 0.131 |
| Árboles usados (early stopping) | **362** (eran 59) | **362** (eran 20) |

**Confiabilidad del número** — lo que el modelo promete contra lo que ocurre. Se
recalcula en cada entrenamiento y queda en `resultados_motor2*.json`:

| tramo | pledge: promete → ocurre | donante: promete → ocurre |
|---|---|---|
| top 1% | 67% → 60% | 67% → 56% |
| top 5% | 46% → 44% | 47% → 40% |
| top 10% | 28% → 27% | 29% → 25% |
| top 20% | 15% → 15% | 16% → 14% |

Del top 5% hacia abajo la probabilidad se puede leer literal. En el top 1%
todavía sobreestima: ahí léase "riesgo muy alto", no un porcentaje exacto.

### Cambios de configuración de septiembre de 2026

| # | Cambio | Por qué |
|---|---|---|
| 1 | Validación **temporal** (últimos 2 meses del train) en vez del último 10% por posición | Ordenar por `pledge_id`/`contact_id` equivale a ordenar por antigüedad: el val quedaba con 5,90% de churn contra 3,48% del test. Cortaba el entrenamiento temprano y sesgaba el calibrador |
| 2 | `boost_from_average=True` | Sin esto el modelo arranca en 0,5 (moneda al aire) en vez de la tasa base. **Solo funciona junto con el cambio 1** — por separado empeora |
| 3 | Tabla de confiabilidad en cada entrenamiento | Guardarraíl: si un reentrenamiento futuro rompe la calibración, queda registrado |
| 4 | Aviso de frescura de datos en `01_motor2_dataset.py` | Si el staging se quedó atrás del snapshot, el riesgo parece altísimo para toda la base. Ver `ORDEN_DE_EJECUCION.md` |

## Cómo ejecutar el pipeline

El pipeline tiene **tres modos de uso** con propósitos y frecuencias distintos:

| Modo | Script | Cuándo | Duración |
|------|--------|--------|----------|
| Re-entrenamiento completo | `run_all.sh` | Cada 6-12 meses, o cuando llegan datos nuevos del CRM | 40–90 min |
| Inferencia mensual (los dos motores) | `run_monthly.sh` | Cada mes, primeros días | 5–20 min |

### Entrenar vs. puntuar

Son dos operaciones distintas y conviene no confundirlas:

- **Entrenar** — el modelo aprende de meses cuyo desenlace ya conocemos. Necesita
  el label, y es lo que permite medirlo. Se hace cada 6-12 meses.
- **Puntuar** — se aplica el modelo guardado al mes en curso, cuyo desenlace aún
  no ocurrió. No necesita label. Se hace cada mes, en segundos.

Puntuar en vez de reentrenar cada mes mantiene dos cosas: las métricas publicadas
siguen siendo las medidas contra un año completo, y los puntajes son comparables
entre meses (un P2 de 0.75 en febrero significa lo mismo que en enero, porque sale
del mismo árbol y del mismo calibrador).

---

### Modo 1 — Re-entrenamiento anual (`run_all.sh`)

Procesa los CSV crudos del CRM desde cero: preprocesamiento → features → entrenamiento de ambos motores → tabla donantes.

#### 1.1 Entorno

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### 1.2 Inputs requeridos

| Carpeta / Archivo | Descripción |
|---|---|
| `raw/*.csv` | Los 9 archivos CSV exportados del CRM de UNICEF Colombia |
| `CUTOFF_DATE` | Fecha de corte: define las ventanas temporales y el año de test de cada motor. Default: `2025-12-31` |

> **Qué NO hace `CUTOFF_DATE`:** no recorta los datos de entrada. Los scripts leen
> todo el staging disponible; el corte solo define ventanas y splits. Si el staging
> llega al `2025-12-31` y pides corte `2025-12-21`, el entrenamiento de Motor 1
> seguirá usando los intentos del 22 al 31 de diciembre. La única excepción es
> `03_timing_reintento_motor1.py`, cuya ventana móvil sí respeta el corte como
> límite superior.

> **Al agregar datos de un año nuevo** hay que registrar los archivos en
> `FILE_GROUPS` (dentro de `00_unify_raw_data.py`): es una lista literal de rutas,
> no un glob. Los archivos que se reemplazan conservando el nombre (Pledge,
> Contact, Allocation — que son snapshots completos del CRM) no requieren cambios.

#### 1.3 Ejecutar

```bash
source .venv/bin/activate
bash scripts/run_all.sh
```

Para cambiar la fecha de corte (p. ej. al incorporar datos de 2026), **`FORCE=1`
es obligatorio**:

```bash
FORCE=1 CUTOFF_DATE=2026-12-31 bash scripts/run_all.sh
```

Sin `FORCE=1` el script salta las etapas cuya salida ya existe y quedarían las
features y modelos del corte anterior. `run_all.sh` avisa si detecta esa
combinación, y `02_train_motor2.py` aborta si el dataset no corresponde al corte
pedido.

Para forzar el re-cálculo de todo desde cero manteniendo el corte por defecto:

```bash
FORCE=1 bash scripts/run_all.sh
```

`run_all.sh` verifica el número de filas/columnas de cada parquet intermedio y produce un resumen PASS/FAIL/SKIP. El log completo queda en `run_all.log`.

---

### Modo 2 — Inferencia mensual, los dos motores (`run_monthly.sh`)

Regenera las tres salidas operativas del mes sin tocar ningún modelo. Conviene
correrlo en los **primeros días del mes**: el snapshot de Motor 2 es el día 1, y
sus features usan todo el historial del mes anterior.

Seis etapas:

| Etapa | Qué hace |
|---|---|
| A | Features incrementales de Motor 1 desde el archivo de pagos YTD |
| B | Timing de cobro/recobro proyectado desde hoy |
| C | Refresca el staging desde `raw/` **solo si le faltan** los pagos del mes que cerró (si está al día, `SKIP`) |
| D | Genera el snapshot del mes para Motor 2 |
| E | Puntúa Motor 2 → lista de retención del mes |
| F | Tabla maestra, tomando el P2 de la etapa E |

**Por qué existe la etapa C:** Motor 1 se alimenta del archivo YTD (que se encadena
sobre las features históricas sin re-procesar 5,3M de filas), pero Motor 2 necesita
el staging, porque una de sus features es `bank_name` y el archivo YTD no la trae.
Si al staging le faltan los pagos del mes que cerró, se regenera (~12 min); si
después de regenerar sigue faltando, el script aborta avisando que el archivo nuevo
no está registrado en `FILE_GROUPS`.

#### 2.1 Inputs requeridos

Todos estos archivos deben existir (se generan con `run_all.sh`):

| Archivo | Descripción |
|---|---|
| `results/motor1/motor1_catboost.cbm` | Modelo Motor 1 entrenado |
| `results/motor1/motor1_calibrador.pkl` | Calibrador de probabilidades |
| `results/motor2/motor2_catboost.cbm` | Modelo Motor 2 entrenado |
| `data/features/motor1_features.parquet` | Features históricas de intentos de cobro |
| `data/features/allocation_features.parquet` | Features de cambios de asignación |
| `data/features/motor2_pledge_month.parquet` | Features mensuales por pledge |
| `results/motor1/test_predictions_motor1.parquet` | Predicciones test Motor 1 |
| `results/motor2/test_predictions_motor2.parquet` | Predicciones test Motor 2 |
| `data/unified/pledge/pledge_unified.parquet` | Datos maestros de pledges |

#### 2.2 Parámetros

| Variable | Qué es | Cuándo cambiarla |
|---|---|---|
| `PAYMENTS_YTD` | Pagos del año en curso (CSV o parquet). Default: `raw/payments_ytd.parquet` | Cada mes: el equipo reemplaza el archivo |
| `RUN_DATE` | Fecha de ejecución ("hoy"). Default: fecha del sistema | Solo para simular resultados en una fecha distinta |
| `CUTOFF_DATE` | **Solo** afecta la tabla maestra: año de agregación de P1/P2 y nombre del archivo de salida. Default: `2025-12-31` | Solo al re-entrenar con nuevos datos |

> La etapa de timing **no** recibe `CUTOFF_DATE` sino `RUN_DATE` como frontera de
> información. Con el corte del modelo, su ventana móvil de 12 meses se quedaría
> anclada en el año del entrenamiento e ignoraría los intentos del año en curso —
> justamente los que trae el archivo YTD.

#### 2.3 Ejecutar

```bash
source .venv/bin/activate
bash scripts/run_monthly.sh
```

Para simular cómo se verían los resultados a fin de mes (sin esperar):

```bash
RUN_DATE=2026-07-31 bash scripts/run_monthly.sh
```

#### 2.4 Outputs generados

| Archivo | Columnas clave | Uso |
|---|---|---|
| `data/features/motor1_features_ytd.parquet` | features del año en curso | Insumo de la etapa de timing |
| `results/motor1/timing_reintento_motor1.csv` | `pledge_id`, `ultimo_intento_ts`, `tipo_ultimo_intento`, `dias_esperar_desde_hoy`, `fecha_recomendada`, `prob_optimo`, `prob_max_absoluto`, `prob_d1`…`prob_d30` | Fecha óptima de cobro para **todos** los donantes activos (último éxito o fallo) |
| `data/entregable/Tabla_Donantes_P1_P2_<CUTOFF_SLUG>.csv` | `pledge_id`, `p1_*` (prob. éxito cobro), `p2_*` (riesgo abandono), `dias_proximo_cobro` / `fecha_proximo_cobro` / `prob_exito_proximo_cobro` (timing óptimo) | Priorizar llamadas de retención a donantes con P2 alto y programar el próximo cobro |

El log completo queda en `run_monthly.log`.

---

### Modo 3 — Puntuar Motor 2 por separado

Las etapas D y E de `run_monthly.sh` también se pueden correr solas, por ejemplo
para refrescar la lista de retención a mitad de mes sin regenerar el resto:

```bash
python3 scripts/02_features_motor2/01_motor2_dataset.py \
    --cutoff-date 2026-02-05 --months 1 \
    --out data/features/motor2_snapshot_2026_02_01.parquet     # ~25 s

# Riesgo por COMPROMISO (pledge)
python3 scripts/03_train/04_score_motor2.py --score-date 2026-02-05 \
    --dataset data/features/motor2_snapshot_2026_02_01.parquet          # ~8 s

# Riesgo por PERSONA (donante) — mismo snapshot, otra unidad
python3 scripts/03_train/04b_score_motor2_donante.py --score-date 2026-02-05 \
    --dataset data/features/motor2_snapshot_2026_02_01.parquet          # ~8 s
```

**Las dos salidas responden preguntas distintas y ambas se generan del mismo
snapshot:**

| Salida | Una fila por | Para qué |
|---|---|---|
| `results/motor2/scoring_motor2_<fecha>.csv` | **compromiso** (`pledge_id`) | qué cobro concreto hay que arreglar; alimenta la tabla maestra |
| `results/motor2_donante/scoring_motor2_donante_<fecha>.csv` | **persona** (`contact_id`) | a quién llamar; es la lista de retención |

Salida pledge-level: `results/motor2/scoring_motor2_2026_02_05.csv` — un pledge por fila con
`prob_churn` (riesgo de cancelar entre el 1 y el 31 de febrero), ordenado de mayor
a menor, más las columnas de contexto que explican el puntaje (`streak_actual`,
`tenure_days`, `days_since_last_attempt`, `n_fallos_30d`). El script imprime el
tamaño de la lista según el umbral (P2 > 0.8, > 0.6, > 0.4, > 0.2) para dimensionar
la campaña, y deja registrado con qué corte se entrenó el modelo que usó.

Requiere que el staging incluya los pagos del mes anterior. Si el snapshot pedido
va por delante de los pagos cargados, el primer comando lo avisa en vez de producir
features congeladas en silencio.

---

### Modo 4 — Ejecución manual paso a paso

El detalle por script (insumos, salidas, filas esperadas) está en
`scripts/ORDEN_DE_EJECUCION.md`. La secuencia del re-entrenamiento completo,
equivalente a `run_all.sh`, es:

```bash
cd v8_entregable
python3 scripts/00_preprocesamiento/00_unify_raw_data.py --raw-dir . --out-dir ./data/unified
python3 scripts/00_preprocesamiento/01_create_staging.py
python3 scripts/01_features_motor1/01_clean_bank_normalization.py
python3 scripts/01_features_motor1/02_enrich_with_contact.py
python3 scripts/01_features_motor1/03_build_allocation_features.py
python3 scripts/02_features_motor2/01_motor2_dataset.py --cutoff-date 2025-12-31
python3 scripts/03_train/01_train_motor1.py --cutoff-date 2025-12-31
python3 scripts/03_train/02_train_motor2.py --cutoff-date 2025-12-31
python3 scripts/03_train/03_timing_reintento_motor1.py --cutoff-date 2025-12-31 --run-date 2026-01-03
python3 scripts/04_features_motor3/01_tabla_donantes_p1_p2.py --cutoff-date 2025-12-31
```

En la etapa de timing, `--cutoff-date` es la frontera de información y `--run-date`
es "hoy" (desde donde se proyectan las fechas). Para una corrida operativa, ambas
deben ser la fecha actual; el `2025-12-31` de arriba corresponde al corte con que
se entrenó este entregable.

### Salidas nuevas en `results/motor1/`

| Archivo | Descripción |
|---------|-------------|
| `motor1_calibrador.pkl` | Calibrador isotónico entrenado sobre el validation set. Corrige el sesgo de probabilidades del modelo y se guarda junto al `.cbm` |
| `test_predictions_motor1.parquet` | Predicciones con dos columnas: `prob_success` (calibrada, usar esta) y `prob_success_cruda` (referencia). Solo parquet — a diferencia de Motor 2, este script no exporta CSV |
| `timing_reintento_motor1.parquet/.csv` | **Timing óptimo de próximo cobro**: una fila por cada donante activo, independientemente de si su último intento fue éxito o fallo. Columnas: `pledge_id`, `ultimo_intento_ts`, `tipo_ultimo_intento` (`exitoso`/`fallo`), `dias_esperar_desde_hoy`, `fecha_recomendada`, `prob_optimo`, y la curva completa `prob_d1` … `prob_d30`. |

## Conteos del pipeline

| Etapa                                   | Filas       | Cols |
| --------------------------------------- | ----------- | ---- |
| Staging `stg_payment_attempts`          | 5.309.073   | 17   |
| `motor1_features`                       | 5.309.073   | 20   |
| `allocation_features`                   | 5.309.073   | 15   |
| `motor2_pledge_month`                   | 2.548.075   | 18   |
| `Tabla_Donantes_P1_P2_<CUTOFF>`         | ~177.883    | 39   |

Las tres primeras filas son independientes de la fecha de corte. `motor2_pledge_month`
y la tabla maestra **sí** cambian con el corte (la ventana de 24 snapshots se mueve),
por eso `run_all.sh` solo valida el conteo de `motor2_pledge_month` cuando el corte
es el default.

## Dependencias

Python 3.11+ con las librerías de `requirements.txt` (`pandas`, `pyarrow`,
`numpy`, `catboost`, `scikit-learn`, `openpyxl`).

---

## Ajustes reunión julio 21

| # | Ajuste | Archivo |
|---|--------|---------|
| 1 | **Candidatos de días diferenciados por tipo de último intento**: fallido usa `[1, 3, 5, 7, 10, 15, 20, 25, 30]`; exitoso usa `[15, 20, 25, 30]` (mínimo 15 días después de un pago exitoso) — ⚠️ **reemplazado** por el ajuste 3 de la reunión del 27 de julio (días 1 a 30 unificados) | `scripts/03_train/03_timing_reintento_motor1.py` |
| 2 | **Tolerancia de 2pp para preferir día más cercano**: si un día anterior al óptimo tiene probabilidad dentro del 2% del máximo, se recomienda ese día más cercano. Nueva columna `prob_max_absoluto` permite ver cuándo se aplicó la tolerancia | `scripts/03_train/03_timing_reintento_motor1.py` |
| 3 | **Todos los pledges activos en el timing**: el output ya no se limita a pledges con fallo activo — incluye todos los donantes activos con su fecha óptima de próximo cobro. Nueva columna `tipo_ultimo_intento` (`exitoso`/`fallo`) | `scripts/03_train/03_timing_reintento_motor1.py` |
| 4 | **Fix: último intento vs. último fallo**: la selección de pledges ahora usa el último intento registrado (no el último fallo). Un pledge que falló en abril pero pagó en mayo no aparece como fallo activo | `scripts/03_train/03_timing_reintento_motor1.py` |

### Motor 1 extendido como optimizador de timing de reintento

El modelo Motor 1 predice si un cobro va a ser exitoso. Se acordó extender su uso para responder una pregunta más accionable para el equipo de cobros: **¿cuándo debería programarse el próximo cobro de cada donante para maximizar la probabilidad de éxito?**

La implementación (`scripts/03_train/03_timing_reintento_motor1.py`) toma el último intento de cada donante activo y simula el modelo en los 30 días candidatos (1 a 30), manteniendo fijas todas las features estructurales y actualizando solo las temporales (día del mes, día de la semana, días desde el último intento, etc.). Los contadores se ajustan según el resultado del último intento: si fue éxito, `pre_failure_streak` se resetea a 0 y `success_count_cum` suma 1; si fue fallo, `failure_count_cum` y `pre_failure_streak` suman 1.

El "último intento de cada donante activo" se busca en una **ventana móvil de 12
meses** que termina en la fecha de corte (`--window-months` cambia el ancho). Antes
se filtraba por año calendario, lo que daba ventanas de ancho variable según el mes
del corte — y vacías al correr en los primeros días de enero.

El output (`results/motor1/timing_reintento_motor1.csv`) tiene una fila por **cada donante activo** — tanto los que fallaron como los que pagaron exitosamente — con la `fecha_recomendada` para el próximo cobro y la columna `tipo_ultimo_intento` para distinguir ambos casos.

---

## Ajustes reunión julio 27

| # | Ajuste | Archivo |
|---|--------|---------|
| 1 | **Regla de 21 días dinámica**: el mínimo de días entre el último intento exitoso y el próximo cobro se calcula desde la fecha del último intento, no desde hoy. `dia_min = max(1, 21 − dias_desde_ultimo)`. Si el donante pagó hace 18 días, solo hay que esperar 3 días más. Esta restricción **no aplica a fallos** — para pledges con último intento fallido, `dia_min = 1`. | `scripts/03_train/03_timing_reintento_motor1.py` |
| 2 | **Tolerancia logarítmica** (reemplaza la tolerancia fija de 2pp): `umbral = 0.016 × log2(d_extra + 1)`. Permite sacrificar hasta ~1.6pp por 1 día de adelanto, ~5.5pp por 10 días y ~7.8pp por 29 días. | `scripts/03_train/03_timing_reintento_motor1.py` |
| 3 | **Candidatos unificados con días 1 a 30**: se agrega desde el día 1 a 30 los candidatos de ambos tipos. Para exitosos, se filtran los días menores a `dia_min` según la regla dinámica. | `scripts/03_train/03_timing_reintento_motor1.py` |

ejecutarlo con

```bash
python3 scripts/03_train/03_timing_reintento_motor1.py \
    --cutoff-date 2025-12-31 --run-date 2026-01-03
```

---

## Ajustes posteriores (fecha de corte y ciclo mensual)

| # | Ajuste | Archivo |
|---|--------|---------|
| 1 | **Fecha de corte configurable en el entrenamiento de Motor 2** (`test = año del corte`), con guard de coherencia contra el dataset, detección de censura del label y modo scoring | `scripts/03_train/02_train_motor2.py` |
| 2 | **`--months` y `--out`** en el dataset de Motor 2, para generar un solo snapshot en un archivo aparte sin tocar el de entrenamiento | `scripts/02_features_motor2/01_motor2_dataset.py` |
| 3 | **Puntuación mensual de Motor 2** sin re-entrenar: carga el `.cbm` + calibrador y puntúa el snapshot del mes | `scripts/03_train/04_score_motor2.py` (nuevo) |
| 4 | **Ventana móvil de 12 meses** en el timing (`--window-months`), en lugar del filtro por año calendario que quedaba vacío al correr en enero | `scripts/03_train/03_timing_reintento_motor1.py` |
| 5 | **Etapa de timing con `RUN_DATE` como frontera de información** (antes recibía el corte del modelo e ignoraba el año en curso) | `scripts/run_monthly.sh` |
| 6 | **Aviso de idempotencia** al cambiar el corte sin `FORCE=1`, y validación de conteos condicionada al corte por defecto | `scripts/run_all.sh` |
| 7 | **Fallback del año de P1**: si el año del corte no tiene intentos cargados (correr en enero), P1 se agrega sobre el último año con datos y lo avisa, en lugar de fallar | `scripts/04_features_motor3/01_tabla_donantes_p1_p2.py` |
| 8 | **`--p2-scoring`** en la tabla maestra: la columna P2 del mes puede venir del scoring del mes en curso en vez del snapshot de entrenamiento. El nombre de la columna se deriva del snapshot real del archivo, no del corte | `scripts/04_features_motor3/01_tabla_donantes_p1_p2.py` |
| 9 | **`run_monthly.sh` cubre los dos motores**: agrega el refresco condicional del staging, el snapshot del mes, la puntuación de Motor 2 y le pasa ese scoring a la tabla maestra | `scripts/run_monthly.sh` |
---


Cualquier consulta técnica sobre estos entregables puede dirigirse al equipo de Iridian.
