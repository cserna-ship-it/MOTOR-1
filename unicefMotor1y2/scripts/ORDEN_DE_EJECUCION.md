# Orden de ejecución de los scripts (v8)

Secuencia exacta del pipeline end-to-end, con insumo y salida de cada script.
La estructura de carpetas es **plana**: todo cuelga de `v8_entregable/`
(`raw/`, `data/`, `results/`, `scripts/`). Los paths de abajo son relativos a
esa raíz.

## Convención de nombres de columnas

Las columnas de los parquets están en `snake_case` porque
`00_unify_raw_data.py` normaliza todos los nombres de los CSV originales. El
mapeo `nombre original ↔ nombre en parquet` está en
`docs/03_Mapeo_columnas_original_vs_trabajo.docx`.

## Secuencia

### Etapa 0 — Preprocesamiento

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `00_preprocesamiento/00_unify_raw_data.py` | `raw/*.csv` (9 archivos) | `data/unified/*.parquet` (5 entidades) |
| 2 | `00_preprocesamiento/01_create_staging.py` | `data/unified/{payments,opportunity,pledge}_unified.parquet` | `data/staging/stg_payment_attempts.parquet` (5.309.073 × 17) |

### Etapa 1 — Features Motor 1 (probabilidad de éxito de cobro)

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `01_features_motor1/01_clean_bank_normalization.py` | `data/staging/stg_payment_attempts.parquet` | `data/features/bank_canonical.parquet` (1 col) |
| 2 | `01_features_motor1/02_enrich_with_contact.py` | staging + `contact_unified.parquet` + bank canonical | `data/features/motor1_features.parquet` (5.309.073 × 20) |
| 3 | `01_features_motor1/03_build_allocation_features.py` | `motor1_features.parquet` + `allocation_unified.parquet` | `data/features/allocation_features.parquet` (5.309.073 × 15) |

### Etapa 2 — Features Motor 2 (probabilidad de churn 30 días)

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `02_features_motor2/01_motor2_dataset.py` | staging + `pledge_unified.parquet` | `data/features/motor2_pledge_month.parquet` (2.548.075 × 19 con el corte por defecto) |
| 2 | `02_features_motor2/01b_motor2_dataset_donante.py` | `motor2_pledge_month.parquet` | `data/features/motor2_donante_month.parquet` — el mismo dataset agregado a `(contact_id, snapshot_ts)` |

**Granularidad:** una fila por `(pledge, snapshot mensual)`. Por cada pledge se
emite un snapshot al primer día de cada mes de la ventana, mientras el pledge
esté vigente (`snapshot >= effective_start` y `snapshot <= npsp_end_date_c`).

**Target:** `label_churn_30d` = 1 si `npsp__EndDate__c` cae en
`(snapshot, snapshot + 30d]`. Depende **solo** de la fecha de cancelación
contractual, no de `payment_status`: un pledge puede fallar muchos cobros sin
churnear si el donante no cancela formalmente.

**Anti-leakage:** todas las features se calculan con información estrictamente
anterior al snapshot (`merge_asof` con `direction="backward"` y
`allow_exact_matches=False`).

**`contact_id` en la salida (18 → 19 columnas).** Desde el cambio a nivel
donante, el script resuelve y adjunta el `contact_id` de cada pledge desde el
staging. Si un pledge apareciera con más de un `contact_id` en su historial de
intentos (error de captura en el CRM), se toma el **más frecuente** — la misma
regla que aplica `motor2_donante_common.py`, para que los dos caminos coincidan.

**Aviso de frescura de datos.** Las features se miden DESDE el snapshot hacia
atrás. Si el staging se quedó meses atrás del snapshot pedido, "días desde el
último intento" le da a todo el mundo un número enorme y el modelo concluye que
la base entera está muerta. Caso real: con el staging hasta 2025-12-31 y snapshot
2026-09-01, la mediana de días sin intento pasa de 31 a 274 y el P2 medio de 0,03
a 0,27. El script avisa cuando el rezago pasa de 45 días:

```
[AVISO] El snapshot más reciente (2026-09-01) va 244 días por delante del último
        pago en el staging (2025-12-31).
        Las features de actividad reciente van a salir infladas y el riesgo va a
        parecer altísimo para toda la base.
```

`run_monthly.sh` ya bloquea este caso (etapa C regenera el staging y aborta si no
alcanza); el aviso cubre la ejecución manual del script.

**Parámetros propios:** `--months N` (cuántos snapshots mensuales generar hacia
atrás desde el mes de corte; default 24) y `--out RUTA` (parquet de salida). Los
acumulados no dependen de la ventana: se calculan siempre sobre toda la historia
de intentos disponible, así que `--months 1` produce exactamente las mismas
features para ese snapshot que `--months 24`. Se usan en la puntuación mensual
(etapa 2.7).

### Etapa 2.5 — Entrenamientos

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `03_train/01_train_motor1.py` | `motor1_features.parquet` + `allocation_features.parquet` | `results/motor1/` (modelo `.cbm`, calibrador `.pkl`, métricas, `test_predictions_motor1.parquet`) |
| 2 | `03_train/02_train_motor2.py` | `motor2_pledge_month.parquet` | `results/motor2/` (modelo `.cbm`, calibrador `.pkl`, métricas, `feature_importance_motor2.csv`, `test_predictions_motor2.parquet` + `.csv`) |
| 3 | `03_train/02b_train_motor2_donante.py` | `motor2_donante_month.parquet` | `results/motor2_donante/` (mismos artefactos, prefijo `_donante`) |

Motor 2 exige que el dataset haya sido generado con la **misma** fecha de corte
(compara el último snapshot contra el mes de corte) y aborta si no coinciden, para
que no se entrene en silencio sobre las features de un corte anterior.

### Etapa 2.5b — Motor 2 a nivel DONANTE

Corre **en paralelo** al Motor 2 pledge-level; no lo reemplaza. Los dos modelos
conviven porque responden preguntas distintas:

| | Unidad | Pregunta que responde |
|---|---|---|
| `results/motor2/` | `(pledge, snapshot)` | ¿este compromiso concreto se cancela este mes? |
| `results/motor2_donante/` | `(donante, snapshot)` | ¿esta persona deja de donar del todo este mes? |

**Regla del label.** Un donante abandona en `(t, t+30d]` si y solo si **todos**
los pledges que tenía activos en `t` cancelan dentro de esa ventana. Como el
label pledge-level ya es 0/1, eso es exactamente el **mínimo** del grupo
`(contact_id, snapshot_ts)`.

**Las 15 features agregadas** (reglas en `scripts/motor2_donante_common.py`):

| Feature | Operación | Por qué |
|---|---|---|
| `n_pledges_activos` | `nunique` | señal nueva: cuántos compromisos tiene el donante |
| `monto_total_mensual` | `sum` | valor económico del donante, no de un pledge aislado |
| `streak_max`, `streak_promedio` | `max`, `mean` | la peor racha de fallos domina el riesgo |
| `tenure_max` / `tenure_min` | `max` / `min` | pledge más viejo vs. más nuevo |
| `cum_intentos/exitos/fallos_total` | `sum` | track record combinado |
| `n_intentos_30d_total`, `n_fallos_30d_total` | `sum` | actividad reciente combinada |
| `success_rate_hist_pond` | ponderado por `cum_intentos` | un pledge con 50 intentos pesa más que uno con 2 |
| `days_since_last_attempt_min` | `min` | actividad más reciente en **cualquiera** de sus pledges |
| `n_bancos_distintos`, `n_metodos_distintos` | `nunique` | diversidad de medios de pago |

**Sin features categóricas.** `last_bank_name` / `last_payment_method` son de
*un* pledge, no del donante; se reemplazaron por conteos de diversidad. El
modelo de donante entrena con `cat_features=[]`.

**Anti-leakage adaptado.** El pledge-level exige que los acumulados sean
monótonos no decrecientes. A nivel donante eso **no** siempre es cierto: si un
donante tenía 2 pledges y uno cancela, ese pledge sale del conjunto activo y
`cum_intentos_total` puede bajar legítimamente. Esa invariante se bajó a
diagnóstico impreso; las que siguen siendo universales
(`days_since_last_attempt_min >= 0`, `tenure_max/min >= 0`) siguen siendo
`assert`.

**Las métricas NO son comparables con las del pledge-level.** El label de
donante tiene menor prevalencia (hay que cancelar *todos* los pledges), así que
la PR-AUC baja mecánicamente aunque el modelo sea igual de bueno. El JSON de
resultados lleva esa advertencia explícita.

Diagnóstico previo opcional (solo imprime, no escribe nada):

```bash
python3 docs/00_diagnostico_donante.py
```

### Configuración de entrenamiento de Motor 2 (septiembre de 2026)

Los dos modelos de Motor 2 (pledge y donante) comparten esta configuración.

**Validación temporal, no posicional.** `scripts/config.py` →
`split_validacion_temporal()`. El validation set (que decide cuándo parar de
entrenar Y contra el que se ajusta el calibrador) son los **últimos
`MESES_VALIDACION = 2` meses del train**, no el último 10% por posición.

Ordenar por `pledge_id` / `contact_id` equivale a ordenar por antigüedad del
donante — los IDs de Salesforce se asignan en orden de creación — así que el
corte posicional dejaba un val con **5,90%** de churn contra **3,48%** del test.
Eso cortaba el entrenamiento en 59 árboles (20 en donante) y sesgaba el
calibrador hacia arriba.

La función imprime las tasas y **avisa si el val no es representativo**:

```
[split] validación = últimos 2 mes(es) del train (desde 2024-11): 201.675 filas (17%)
[split] tasa del evento — train 0.0454 | val 0.0387 | último mes 0.0408
```

> La ventana de **2 meses está afinada contra el corte 2025-12-31**, no es
> universal: con 3 meses entra el pico de octubre-2024 y la calibración vuelve a
> degradarse. Al mover la fecha de corte, mirar el `[AVISO]` y re-correr
> `docs/06_experimento_calibracion_motor2.py`.

**`boost_from_average=True`.** Es el default de CatBoost para RMSE pero **no**
para `Logloss`. Sin él el modelo arranca prediciendo 0,5 en vez de la tasa base.
**Solo funciona junto con la validación temporal** — activado sobre el split
posicional el modelo para en 0-3 árboles y la PR-AUC cae a 0,296.

**Tabla de confiabilidad en cada entrenamiento.** `config.py` →
`tabla_confiabilidad()`. Compara lo que el modelo promete contra lo que ocurre
en el top 1/5/10/20%, se imprime en el log y se guarda en
`resultados_motor2*.json` → `catboost.confiabilidad_tramo_alto`. Es el
guardarraíl: si un reentrenamiento futuro rompe la calibración, queda ahí.

Justificación completa y experimento reproducible: `docs/07_Informe_calibracion.md`
y `docs/06_experimento_calibracion_motor2.py`.

### Etapa 2.6 — Timing óptimo de reintento (Motor 1)

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `03_train/03_timing_reintento_motor1.py` | `motor1_features.parquet` + `allocation_features.parquet` + `motor1_catboost.cbm` + `motor1_calibrador.pkl` | `results/motor1/timing_reintento_motor1.parquet` + `.csv` |

**Qué genera:** una fila por **cada donante activo** — tanto los que su último
intento fue exitoso como los que fue fallido. Para cada uno simula el modelo en
los 30 días candidatos (1 a 30) y entrega el día óptimo para el próximo cobro.

Reglas de negocio aplicadas (acuerdos de las reuniones del 21 y 27 de julio):

- **Regla de 21 días dinámica** (solo para último intento exitoso):
  `dia_min = max(1, 21 − dias_desde_ultimo)`. Si el donante pagó hace 18 días,
  solo faltan 3. Para fallos no aplica: `dia_min = 1`.
- **Tolerancia logarítmica:** `umbral = 0.016 × log2(d_extra + 1)`. Si un día
  anterior al óptimo queda dentro del umbral, se recomienda el más cercano.
  `prob_max_absoluto` deja ver cuándo se aplicó.

**Parámetros:** `--cutoff-date` (frontera de información: hasta cuándo se conocen
los intentos), `--run-date` ("hoy": desde aquí se proyectan las fechas),
`--window-months` (ancho de la ventana de historia, default 12).

**Columnas de salida:**

| Columna | Descripción |
|---------|-------------|
| `pledge_id` | Identificador del donante |
| `ultimo_intento_ts` | Fecha del último intento (exitoso o fallido) |
| `tipo_ultimo_intento` | `exitoso` / `fallo` |
| `dias_desde_ultimo_hoy` | Días entre el último intento y `--run-date` |
| `dias_esperar_desde_hoy` | Días a esperar desde `--run-date` |
| `fecha_recomendada` | **Fecha concreta** del próximo cobro (`run_date + dias_esperar`) |
| `prob_optimo` | Probabilidad de éxito en el día recomendado |
| `prob_max_absoluto` | Probabilidad del máximo absoluto (≠ `prob_optimo` si actuó la tolerancia) |
| `prob_d1` … `prob_d30` | Curva completa de probabilidades por día simulado |

### Etapa 2.7 — Puntuación mensual de Motor 2 (no entrena)

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `03_train/04_score_motor2.py` | `motor2_catboost.cbm` + `motor2_calibrador.pkl` + `data/features/motor2_snapshot_<AAAA_MM_01>.parquet` | `results/motor2/scoring_motor2_<AAAA_MM_DD>.parquet` + `.csv` |

Esta etapa **no forma parte del re-entrenamiento anual** (`run_all.sh` no la
ejecuta): es la corrida mensual que responde "quién va a cancelar este mes".
Ver "Entrenar vs. puntuar (Motor 2)" más abajo.

**Columnas de salida:** `pledge_id`, `snapshot_ts`, `prob_churn` (calibrada — usar
esta), `prob_churn_cruda`, más columnas de contexto que explican el puntaje
(`streak_actual`, `tenure_days`, `days_since_last_attempt`, `n_fallos_30d`,
`n_intentos_30d`, `success_rate_hist`, `last_payment_amount`,
`last_payment_method`, `last_bank_name`). Ordenado por `prob_churn` descendente.

No se emite el label: el mes que se puntúa todavía no tiene desenlace.

**Versión a nivel donante:** `03_train/04b_score_motor2_donante.py`, mismos
argumentos (`--score-date`, `--dataset`). Agrega el snapshot pledge-level a
nivel donante usando `scripts/motor2_donante_common.py` — el **mismo** módulo
que construyó el dataset de entrenamiento, para que la agregación y la
imputación no se puedan desincronizar — y escribe
`results/motor2_donante/scoring_motor2_donante_<AAAA_MM_DD>.parquet` + `.csv`.
Requiere que el snapshot traiga `contact_id` (lo trae desde la etapa 2).

### Etapa 3 — Tabla intermedia Motor 3 (P1 + P2 por donante)

| # | Script | Insumo | Salida |
|---|--------|--------|--------|
| 1 | `04_features_motor3/01_tabla_donantes_p1_p2.py` | predicciones M1 + M2 + features M1 + allocation features + `timing_reintento_motor1.parquet` + `pledge_unified.parquet` | `data/entregable/Tabla_Donantes_P1_P2_<CUTOFF>.parquet` (~177.883 × 39) |

**De dónde sale cada probabilidad:** P1 se agrega desde
`test_predictions_motor1.parquet` (intentos del año de corte) y P2 desde
`test_predictions_motor2.parquet`, o sea del **snapshot con que se entrenó** el
modelo — no del scoring mensual de la etapa 2.7. La columna del mes de corte se
llama `p2_<mmm>_<aaaa>` con el mes en inglés (`p2_dec_2025`, `p2_jan_2026`).

Si el año de corte no tiene intentos de cobro cargados, el script aborta con un
mensaje explícito (P1 no se puede calcular) en vez de fallar en pandas.

## Cambios funcionales de v8 (Motor 1)

Todos localizados en `03_train/01_train_motor1.py`, salvo el último:

1. **`sample_frac = 1.0`** — se entrena con el 100% del train (2022-2024).
2. **Orden por pledge dentro del script** — tras cargar cada slice se re-ordena
   por `(pledge_id, attempt_ts, payment_id)`. Necesario porque `df.sample()`
   baraja las filas; sin el re-orden el holdout de validación sería aleatorio.
3. **`auto_class_weights = "Balanced"`** — balanceo automático de clases.
4. **Validación anti-leakage en el mismo script** (`validate_temporal_integrity`)
   — comprueba que no haya `days_since_last_attempt` negativos y que el primer
   intento de cada pledge llegue con historial acumulado en cero (evidencia del
   `shift(1)` upstream). Aborta si falla.
5. **`payment_method` en lugar de `payment_submethod`** — el medio de pago es la
   feature categórica; `payment_submethod` está frecuentemente vacío. Se propaga
   desde `02_enrich_with_contact.py` (agrega `payment_method` a `motor1_features`).
6. **Fix de días negativos** (`04_features_motor3/01_tabla_donantes_p1_p2.py`) —
   `tenure_days_at_<CUTOFF_SLUG>` se ancla a `min(pledge_start_date, primer intento
   observado en toda la historia)`, eliminando los tenures negativos que producía
   `npsp__Start_Date__c` con fechas futuras.

Los cambios funcionales de **Motor 2** (sin balanceo de clases, calibración
isotónica, corte configurable, censura del label, modo scoring, puntuación mensual)
están documentados con su justificación en `README.md` → "Ajustes del Motor 2".

## Fecha de corte configurable

Todo el pipeline lee la fecha de corte de `scripts/config.py`, que la resuelve en
este orden: `--cutoff-date` (CLI) > `CUTOFF_DATE` (entorno) > default `2025-12-31`.

```bash
# Re-entrenar todo con otro corte (FORCE=1 es obligatorio: ver nota abajo)
FORCE=1 CUTOFF_DATE=2026-01-03 bash scripts/run_all.sh
```

Qué significa el corte en cada motor:

| Script | Efecto de la fecha de corte |
|---|---|
| `02_features_motor2/01_motor2_dataset.py` | Ventana de 24 snapshots mensuales terminando en el **mes** del corte. Corte `2026-01-03` → snapshots `2024-02` … `2026-01` |
| `03_train/01_train_motor1.py` | `test = intentos del año del corte`; `train = 2022 … año-1` |
| `03_train/02_train_motor2.py` | `test = snapshots del año del corte`; `train = snapshots de años <= año-1` |
| `03_train/03_timing_reintento_motor1.py` | Fin de la ventana móvil de historia: se busca el último intento de cada pledge en `(corte - 12 meses, corte]`. El ancho se cambia con `--window-months` |
| `03_train/04_score_motor2.py` | No usa `--cutoff-date` sino `--score-date` ("hoy"): puntúa el snapshot del primer día de ese mes |
| `04_features_motor3/01_tabla_donantes_p1_p2.py` | Año de agregación de P1/P2 y nombre de las columnas (`p2_jan_2026`, `tenure_days_at_2026_01_03`) y del archivo de salida |

**Qué NO hace el corte:** no recorta los datos de entrada. Define ventanas y
splits, pero los scripts leen todo el staging disponible. Si el staging llega al
`2025-12-31` y se pide corte `2025-12-21`, el entrenamiento de Motor 1 seguirá
usando los intentos del 22 al 31 de diciembre. La única excepción es
`03_timing_reintento_motor1.py`, cuya ventana móvil sí respeta el corte como
límite superior.

**Nota de idempotencia:** `run_all.sh` salta las etapas cuya salida ya existe. Al
cambiar el corte hay que correr con `FORCE=1`, o las features/modelos del corte
anterior quedarían en su lugar. `02_train_motor2.py` detecta esa inconsistencia
(compara el último snapshot del dataset contra el mes de corte) y aborta con
instrucciones en vez de entrenar sobre el corte equivocado.

**Límite por datos disponibles:** el corte no puede ir por delante de los pagos
cargados. Con datos hasta `2025-12-31`, un corte en 2026 permite **puntuar**
Motor 2 (el snapshot `2026-01-01` usa solo historia previa), pero Motor 1 y la
tabla maestra necesitan intentos de cobro de 2026: sin ellos,
`01_tabla_donantes_p1_p2.py` aborta indicando que falta cargar los pagos del año.

**Modo scoring de Motor 2:** si el test cae en el mes en curso, su ventana de 30
días todavía no cerró — como churn solo aparecen las cancelaciones ya registradas
en el CRM. El script lo detecta (`snapshot + 30d > corte`), marca
`corte.n_test_label_censurado` en `resultados_motor2.json` y avisa que esas
métricas son optimistas; si el test queda con una sola clase, omite las métricas
y solo escribe las predicciones.

## Entrenar vs. puntuar (Motor 2)

Son dos operaciones con distinta periodicidad:

| | Script | Cada cuánto | Qué necesita |
|---|---|---|---|
| **Entrenar** | `03_train/02_train_motor2.py` | cada 6-12 meses | un período cerrado con desenlace conocido, para aprender y para medirse |
| **Puntuar** | `03_train/04_score_motor2.py` | cada mes | solo el snapshot del mes en curso; no necesita label |

Puntuar con el modelo guardado (en vez de reentrenar cada mes) mantiene dos cosas:
las métricas publicadas siguen siendo las medidas contra un año completo, y los
puntajes son comparables entre meses — un P2 de 0.75 en febrero significa lo mismo
que en enero porque sale del mismo árbol y del mismo calibrador.

**La forma normal de correr el ciclo mensual es `run_monthly.sh`** (ver más abajo),
que orquesta todo esto. Los comandos individuales sirven para correr una etapa
suelta o para entender qué hace el script:

```bash
# 1. Refrescar el staging con los pagos del mes que cerró
#    (agregar el CSV nuevo a raw/ y a FILE_GROUPS en 00_unify_raw_data.py)
python3 scripts/00_preprocesamiento/00_unify_raw_data.py --raw-dir . --out-dir data/unified
python3 scripts/00_preprocesamiento/01_create_staging.py

# 2. Generar SOLO el snapshot del mes (segundos)
python3 scripts/02_features_motor2/01_motor2_dataset.py \
    --cutoff-date 2026-02-05 --months 1 \
    --out data/features/motor2_snapshot_2026_02_01.parquet

# 3. Puntuar → lista de llamadas de retención
python3 scripts/03_train/04_score_motor2.py --score-date 2026-02-05

# 4. Fechas de cobro/recobro (Motor 1, tampoco reentrena)
python3 scripts/03_train/03_timing_reintento_motor1.py \
    --cutoff-date 2026-02-05 --run-date 2026-02-05

# 5. Tabla maestra con el P2 del mes en curso
python3 scripts/04_features_motor3/01_tabla_donantes_p1_p2.py \
    --cutoff-date 2026-02-05 \
    --p2-scoring results/motor2/scoring_motor2_2026_02_05.parquet
```

En los pasos 2-5 la fecha es la misma —hoy— y ninguno reentrena: todos cargan el
modelo guardado. El corte actúa como frontera de información ("esto es lo que sé
hoy"), no como split de entrenamiento.

Salida: `results/motor2/scoring_motor2_2026_02_05.csv` — un pledge por fila con
`prob_churn` (riesgo de cancelar entre el 1 y el 31 de febrero) más las columnas
de contexto que explican el puntaje (`streak_actual`, `tenure_days`,
`days_since_last_attempt`, `n_fallos_30d`).

El paso 1 es el costoso (~12 min); los pasos 2 y 3 tardan ~30 segundos juntos. Si
el snapshot pedido va por delante de los pagos cargados, el paso 2 avisa
explícitamente en vez de producir features congeladas en silencio.

**Sobre el paso 5:** sin `--p2-scoring`, la columna P2 de la tabla saldría del
snapshot con que se entrenó el modelo — un mes ya cerrado. Con el parámetro sale
del scoring del mes en curso, y el nombre de la columna se deriva del snapshot que
trae ese archivo (`p2_feb_2026`), no del corte.

## Ejecución mensual con `run_monthly.sh` (los dos motores)

`scripts/run_monthly.sh` cubre el ciclo mensual completo sin re-entrenar: las seis
etapas cargan los modelos ya guardados.

```bash
# Ejecución normal (usa la fecha de hoy como RUN_DATE)
PAYMENTS_YTD=raw/payments_2026_ytd.csv bash scripts/run_monthly.sh

# Simular una fecha de ejecución distinta
PAYMENTS_YTD=raw/payments_2026_ytd.csv RUN_DATE=2026-02-05 bash scripts/run_monthly.sh
```

**Parámetros:**

| Variable | Descripción | Cuándo cambiarla |
|---|---|---|
| `PAYMENTS_YTD` | Pagos del año en curso (CSV o parquet). Default `raw/payments_ytd.parquet` | Cada mes (el equipo reemplaza el archivo) |
| `RUN_DATE` | "Hoy". Define desde dónde se proyectan las fechas de cobro **y** cuál snapshot de Motor 2 se puntúa (el día 1 de ese mes). Default: fecha del sistema | Solo para simular otra fecha |
| `CUTOFF_DATE` | Corte con que se entrenó el modelo. **Solo** nombra la tabla maestra y define el año de agregación de P1. Default `2025-12-31` | Solo al re-entrenar |
| `REFRESH_STAGING` | `auto` (default): regenera el staging solo si le faltan datos. `1` siempre, `0` nunca | Rara vez |

> La etapa de timing **no** recibe `CUTOFF_DATE`, recibe `RUN_DATE` como frontera
> de información. Con el corte del modelo su ventana móvil de 12 meses se quedaría
> anclada en el año del entrenamiento e ignoraría los intentos del año en curso.

**Etapas y outputs:**

| Etapa | Salida |
|---|---|
| A · Features incrementales YTD (Motor 1) | `data/features/motor1_features_ytd.parquet` |
| B · Timing de reintento | `results/motor1/timing_reintento_motor1.parquet` + `.csv` — una fila por **cada donante activo** (último intento exitoso o fallido) |
| C · Staging al día | Regenera `data/staging/` desde `raw/` solo si le faltan los pagos del mes que cerró (~12 min); si ya está al día, `SKIP` |
| D · Snapshot del mes (Motor 2) | `data/features/motor2_snapshot_<AAAA_MM_01>.parquet` |
| E · Puntuación de Motor 2 | `results/motor2/scoring_motor2_<AAAA_MM_DD>.parquet` + `.csv` — lista de retención del mes |
| E2 · Puntuación de Motor 2 **donante** | `results/motor2_donante/scoring_motor2_donante_<AAAA_MM_DD>.parquet` + `.csv`. Se salta con `SKIP` (sin fallar) si el modelo de donante no está entrenado |
| F · Tabla maestra | `data/entregable/Tabla_Donantes_P1_P2_<CUTOFF_SLUG>.parquet` + `.csv`, con la columna P2 tomada del scoring de la etapa E |

**Por qué hay dos caminos de datos.** Motor 1 se alimenta del archivo YTD, que se
encadena sobre las features históricas sin re-procesar 5,3M de filas. Motor 2
necesita el staging, porque una de sus features es `bank_name` y el archivo de
features YTD no la trae. De ahí la etapa C: si al staging le faltan los pagos del
mes que cerró, se regenera desde `raw/`. Si después de regenerar sigue faltando,
el script aborta indicando que el archivo nuevo no está registrado en
`FILE_GROUPS`.

## Pruebas

```bash
python3 tests/test_motor2_donante.py
```

26 pruebas sintéticas de la lógica de agregación pledge → donante: la regla del
label (mínimo del grupo), cada regla de agregación, la resolución de `contact_id`
ambiguo, y la paridad de imputación entre entrenamiento y puntuación. **No
necesita datos reales ni artefactos del pipeline** — construye los casos a mano.

## Dependencias

Python 3.11+ con las librerías de `requirements.txt` (raíz del entregable).
