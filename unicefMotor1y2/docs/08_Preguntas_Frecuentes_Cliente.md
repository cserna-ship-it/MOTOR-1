# Preguntas frecuentes del cliente — Motor 2

**UNICEF Colombia · septiembre de 2026**

Cada pregunta con el archivo que la respalda y una explicación sin tecnicismos.
Las cifras son las vigentes (corte 2025-12-31, después de la corrección de
calibración de septiembre — ver `07_Informe_calibracion.md`).

**Índice**

| # | Pregunta |
|---|---|
| [1](#1) | ¿Qué cambió en las features al pasar de pledge a donante? |
| [2](#2) | ¿El CatBoost tiene `Balanced`? |
| [3](#3) | Sin `Balanced`, ¿cómo se traduce lo que dice el modelo a un número real? |
| [4](#4) | ¿Cómo funciona el calibrador, sin que parezca magia? |
| [5](#5) | Sin `Balanced`, ¿el modelo nunca puede decir 0,80? ¿por qué hay un crudo de 0,93? |
| [6](#6) | La tabla del top 1/5/20% no se entiende: dice 67% pero en el CSV veo 0,97 |
| [7](#7) | Si el máximo es ~60%, ¿estamos llamando a gente que no está en riesgo? |
| [8](#8) | ¿Cómo maneja negocio el top 1%? ¿tiene que restar puntos? |
| [9](#9) | ¿Cómo se comprobaron los resultados? |
| [10](#10) | ¿Con qué fecha se corrieron las pruebas? ¿hacen falta más datos? |
| [11](#11) | ¿Cómo corro la simulación de "quién se me va este mes"? |
| [12](#12) | ¿Motor 2 acepta la fecha de ejecución como parámetro? ¿cuál es el comando? |
| [13](#13) | `days_since_last_attempt_min` ¿contra qué fecha se calculó? |
| [14](#14) | ¿Cómo ve el equipo la salida? |
| [15](#15) | ¿Valdría la pena incluir el monto de los pledges como feature? |

---

<a id="1"></a>
## 1. ¿Qué cambió en las features al pasar de pledge a donante?

**Respaldo:** `scripts/motor2_donante_common.py` (líneas 44-68 la lista, 145-200
las reglas de agregación).

El modelo viejo miraba **un compromiso de pago**. El nuevo mira **una persona**,
que puede tener varios. Hubo que decidir, para cada dato, cómo se combinan los
compromisos de una misma persona. **No fue un promedio: cada dato usa la regla
que tiene sentido de negocio.**

**14 features (pledge) → 15 (donante).**

| Dato | Cómo se combina | Por qué así |
|---|---|---|
| `n_pledges_activos` | cuenta | **Dato nuevo.** Antes no existía "cuántos compromisos tiene esta persona" |
| `monto_total_mensual` | suma | Lo que aporta la persona, no un recibo suelto |
| `streak_max` | **máximo** | Si uno de sus pagos lleva 20 rechazos, esa persona está en problemas — aunque los otros vayan bien. El promedio lo escondería |
| `streak_promedio` | promedio | Distingue "uno malo entre varios buenos" de "todos malos" |
| `tenure_max` / `tenure_min` | máx / mín | Antigüedad del compromiso más viejo vs. el más nuevo |
| `cum_intentos/exitos/fallos_total` | suma | Historial combinado |
| `n_intentos_30d_total`, `n_fallos_30d_total` | suma | Actividad del último mes |
| `success_rate_hist_pond` | **ponderado** | Un compromiso con 50 cobros pesa más que uno con 2 |
| `days_since_last_attempt_min` | **mínimo** | Si tiene uno muerto y otro vivo, importa el vivo |
| `n_bancos_distintos`, `n_metodos_distintos` | cuenta distintos | Diversidad de medios de pago |

**Lo que se perdió (hay que decirlo).** El modelo pledge usaba tres datos de
texto: **banco, método de pago y pasarela**. Son de *un* compromiso, no de la
persona, así que se reemplazaron por conteos de diversidad. **El modelo de
donante no usa ninguna variable categórica.**

**Y el cambio de fondo no fue una feature, fue la definición del evento.** Antes:
*"¿se cancela este compromiso?"*. Ahora: *"¿esta persona deja de donar del
todo?"*. Una persona con dos compromisos que cancela uno y mantiene el otro **no
se fue** — el modelo viejo la contaba como fuga.

> **Dato de realidad:** en los datos de UNICEF solo el **0,35%** de los
> donante-mes tiene más de un compromiso activo a la vez. El cambio es
> conceptualmente correcto pero mueve poco. Detalle en
> `05_Integracion_Motor2_Donante.md` §6.

---

<a id="2"></a>
## 2. ¿El CatBoost tiene `Balanced`?

**Respaldo:** `scripts/03_train/02_train_motor2.py` líneas 334-360 ·
`docs/04_Informe_auto_class_weights.md`

**Motor 2: NO. Motor 1: SÍ.** No es descuido, es una decisión medida.

**Qué es.** Solo el 3,3% de los donantes se va. Si el modelo dijera "nadie se va"
acertaría el 96,7% — y sería inútil. `Balanced` le dice *"préstale más atención a
los casos raros"*, multiplicando artificialmente el peso de las fugas.

**Por qué en Motor 2 se quitó** (`docs/02b_experimento_motor2.py`):

| | con `Balanced` | sin `Balanced` |
|---|---|---|
| AUC | 0,878 | **0,896** |
| PR-AUC | 0,281 | **0,367** (+30%) |
| árboles antes de parar | 3-4 | 55-60 |

Con `Balanced` el modelo se saturaba en 3 o 4 árboles: dejaba de aprender casi de
inmediato.

**Por qué en Motor 1 sí se deja.** Motor 1 predice si un cobro sale bien, y ahí el
desbalance es otro (39% de éxito, no 3%). **La misma herramienta ayuda en un
motor y estorba en el otro** — por eso se probó cada uno por separado.

---

<a id="3"></a>
## 3. Sin `Balanced`, ¿cómo se traduce lo que dice el modelo a un número real?

**Respaldo:** `scripts/03_train/02_train_motor2.py` líneas 370-390 ·
`docs/07_Informe_calibracion.md`

**Hay dos columnas en el CSV y hay que usar la correcta:**

- `prob_churn_donante_cruda` → **no usar.** Número interno del modelo.
- `prob_churn_donante` → **esta.** Ya pasó por el calibrador.

Traducción real del calibrador de donante (`motor2_donante_calibrador.pkl`):

| el modelo crudo dice | el calibrador traduce a |
|---|---|
| 0,02 | 1,6% |
| 0,10 | 25,7% |
| 0,20 | 34,9% |
| 0,30 | **39,8%** |
| 0,40 | 50,6% |
| 0,96 | 100% |

> **Corrige un malentendido común:** *"si el modelo dice 0,3 eso equivale a 90%"*
> era cierto **antes** de la corrección de septiembre. Ya no. Hoy 0,30 crudo es
> ~40%, y la columna calibrada se lee directo.

**Qué tan bien traduce** — se recalcula en cada entrenamiento y queda en
`results/motor2_donante/resultados_motor2_donante.json`:

| grupo | promete | ocurrió | diferencia |
|---|---|---|---|
| top 1% | 67% | 56% | +11 pts |
| top 5% | 47% | 40% | +7 pts |
| top 10% | 29% | 25% | +4 pts |
| top 20% | 16% | 14% | +2 pts |

**Del top 5% hacia abajo se lee literal.** Antes de la corrección esas
diferencias eran +34, +23 y +11 puntos.

---

<a id="4"></a>
## 4. ¿Cómo funciona el calibrador, sin que parezca magia?

Son **tres piezas en fila**, no una:

```
features → [CatBoost] → puntaje crudo → [calibrador] → probabilidad
                            0,93                          0,97
```

**Cómo se construye la tabla de traducción** — esto es literalmente todo:

1. Se aparta un grupo de donantes **cuyo desenlace ya se conoce** (nov-dic 2024).
2. Se ordenan por el puntaje que les dio el modelo.
3. Se agrupan en bloques.
4. En cada bloque se cuenta: *de estos 500 que el modelo puntuó ~0,30, ¿cuántos
   se fueron de verdad?* → 199 → **39,8%**.
5. Se guarda la correspondencia: `0,30 → 0,398`.

Es una tabla de conversión **medida contra la realidad**, no una fórmula.

**"Isotónica"** significa que se le impone una sola regla: la tabla **nunca puede
bajar**. Si el modelo puntúa más alto a A que a B, la probabilidad traducida de A
nunca queda por debajo de la de B. Por eso el calibrador **no cambia el orden** —
solo reetiqueta la escala. El AUC es idéntico antes y después.

Son ~4 líneas de código, y el archivo `.pkl` guardado es esa tabla.

---

<a id="5"></a>
## 5. Sin `Balanced`, ¿el modelo nunca puede decir 0,80?

**Correcto para el promedio. Incorrecto para los individuos.** Medido sobre el
test 2025 (tasa base 3,29%):

```
promedio de todos los puntajes crudos = 0,0340   ← ≈ la tasa base ✓
pero van desde 0,0005 hasta 0,9707               ← aquí falla la intuición

percentil 50:   0,0079
percentil 99:   0,6605
percentil 99,9: 0,9072

1.626 donantes con puntaje crudo > 0,90
```

**Por qué puede decir 0,93.** El modelo no aprende *"¿qué porcentaje de la base se
va?"*. Aprende *"dado ESTE donante, ¿qué fracción de los que se le parecen se
fue?"*. Si entre los donantes con 17 cobros rechazados seguidos + 34 días sin
actividad históricamente se fue el 93%, aprende 0,93 para ese perfil. **La tasa
base ancla el promedio de la población, no cada caso.**

Es como decir "el 3% de los colombianos vive en Cartagena" — eso no impide
afirmar con 95% de certeza que alguien que acaba de comprar tiquete a Cartagena
va para allá.

**El crudo ya nace casi correcto:**

| el crudo dice | se fueron de verdad |
|---|---|
| 0,5% | 0,5% ✓ |
| 2,3% | 2,5% ✓ |
| 39,2% | 37,8% ✓ |
| 86,9% | 62,1% ← solo el extremo se desvía |

**Ese es el argumento para quitar `Balanced`:** `Balanced` le miente al modelo
diciéndole que las fugas son tan comunes como las permanencias, y eso empuja
*todos* los números hacia arriba hasta que dejan de significar nada.

**Y el malentendido que se arrastraba** no era por quitar `Balanced`, era el
modelo a medio entrenar:

| | máximo del puntaje crudo |
|---|---|
| antes de la corrección (20 árboles) | **0,574** ← nunca podía decir "muy alto" |
| después (362 árboles) | **0,971** |

---

<a id="6"></a>
## 6. La tabla del top 1/5/20% no se entiende

**El "67%" es el PROMEDIO del grupo, no un donante.**

```
El top 1% son 13.587 donantes
Sus probabilidades van de 0,506 a 1,000
El promedio del grupo es 0,667   ← ese es el "67%"
Del grupo se fue el 56,5%
```

Por eso en el CSV se ven donantes con 0,97: **están dentro del top 1%, en la
parte alta del rango.** La tabla resume el grupo; el CSV muestra cada persona.

---

<a id="7"></a>
## 7. Si el máximo es ~60%, ¿estamos llamando a gente que no está en riesgo?

**No — ese 56% era el promedio del grupo.** Mirando individualmente el extremo:

| donantes con probabilidad ≥ | cuántos | **se fueron de verdad** | lift |
|---|---|---|---|
| 0,95 | 426 | **83,6%** | 25x |
| 0,90 | 2.580 | 68,1% | 21x |
| 0,80 | 3.426 | 62,7% | 19x |

**Un donante en 0,97 se va el 83% de las veces.** No se está llamando a gente sin
riesgo: se está llamando a gente que se va 25 veces más que el promedio.

Del otro lado: **el 91% de la base tiene probabilidad < 0,05, y de esos se fue el
0,94%.** El modelo despeja con confianza a la gran mayoría.

*(El tramo ≥0,99 muestra 73,5%, más bajo que ≥0,95. Son 102 donantes: es ruido,
no una inversión real.)*

---

<a id="8"></a>
## 8. ¿Cómo maneja negocio el top 1%? ¿tiene que restar puntos?

**No deben restar nada.** Pedirle a retención aritmética mental es garantía de
error. Dos opciones:

1. **(Recomendada)** Agregar una columna `segmento_riesgo` al CSV con la etiqueta
   ya calculada — `CRÍTICO / ALTO / MEDIO / BAJO` — más `se_van_historicamente`
   con el % observado de ese segmento. Retención nunca ve un número que tenga que
   interpretar. *No implementado todavía: ~15 líneas en los dos scripts de
   scoring.*
2. Entregar la tabla de conversión al lado del CSV (ya se genera sola en el JSON
   de cada entrenamiento) como referencia.

---

<a id="9"></a>
## 9. ¿Cómo se comprobaron los resultados?

**Respaldo:** `scripts/run_all.sh` · `tests/test_motor2_donante.py` ·
`docs/06_experimento_calibracion_motor2.py`

Seis controles independientes, todos automáticos:

1. **Conteos exactos en cada paso.** `run_all.sh` verifica filas y columnas de
   cada archivo intermedio. Los 6 de preprocesamiento dieron exacto (5.309.073
   intentos, 464.807 compromisos). Si un CSV llega mal, el pipeline se detiene.

2. **Prueba de "no hacer trampa" (anti-leakage).** El riesgo más grave es que el
   modelo vea información del futuro: se ve buenísimo en pruebas y falla en
   producción. El código **aborta** ante días negativos, antigüedad negativa o
   contadores que retroceden. Validado sobre 321.620 compromisos.

3. **Prueba con el futuro tapado.** Se entrena **solo con 2024** y se evalúa
   **solo con 2025**, un año que nunca vio.

4. **Comparación contra reglas simples.**

   | | AUC |
   |---|---|
   | tirar una moneda | 0,500 |
   | regla "cuántos rechazos seguidos lleva" | 0,845 |
   | regla "qué tan antiguo es" | 0,654 |
   | **el modelo** | **0,898** |

5. **26 pruebas con casos inventados a mano** (`tests/test_motor2_donante.py`).
   Ejemplo: un donante con dos compromisos donde uno cancela y otro no debe dar
   "no se fue". Corre con `python3 tests/test_motor2_donante.py`, sin datos
   reales.

6. **Prueba de que producción calcula igual que el entrenamiento.** El error más
   peligroso es que el proceso mensual difiera apenas del entrenamiento: nadie se
   entera y las predicciones salen torcidas. Se puntuó un mes por las dos vías:

   ```
   126.509 donantes · diferencia máxima: 0.000
   ```

   Además ambos comparten literalmente el mismo archivo de código
   (`motor2_donante_common.py`), así que **no pueden** separarse.

---

<a id="10"></a>
## 10. ¿Con qué fecha se corrieron las pruebas? ¿hacen falta más datos?

**Respaldo:** `scripts/config.py` · `run_all.sh` · `run_monthly.sh`

### Lo primero: NO hacen falta más datos

Los datos que hay alcanzan de sobra para entrenar al corte 2026-01-03. **Ya se
corrió**: 2.451.314 filas de entrenamiento, sin nada faltante.

La fecha de corte **no es "hasta dónde deben llegar los datos"**. Es *"en qué día
me estoy parando para mirar hacia atrás"*. Parado el 3 de enero de 2026, se mira
hacia atrás a los pagos hasta el 31 de diciembre de 2025 — justo lo que hay.

### Las dos fuentes de datos NO llegan hasta la misma fecha

Esto es lo que hace que todo funcione, y no es obvio:

| Fuente | Hasta cuándo llega | Para qué sirve |
|---|---|---|
| **Pagos** (`payments`) | 2025-12-31 | construir las **features** (rachas, montos, días sin cobro) |
| **Cancelaciones** (`pledge`) | **2026-03-16** | saber el **desenlace**: quién se fue |

Las dos exportaciones del CRM se tomaron en momentos distintos. Por eso enero de
2026 se puede **puntuar** (features de los pagos de diciembre) *y además*
**calificar** (desenlaces del export de marzo).

### Entonces, ¿por qué dos fechas?

No es que el sistema necesite dos. Es **un solo pipeline con un solo parámetro**,
corrido dos veces sobre los mismos datos, porque hay un dilema:

> **Un período puede ENSEÑARLE al modelo o CALIFICARLO, no las dos cosas.**

| Corte | Qué hace | Consecuencia |
|---|---|---|
| **2025-12-31** | aparta todo 2025 como examen; entrena solo con 2024 | se puede calificar, pero el modelo estudió 1 año |
| **2026-01-03** | entrena con 2024 **y todo 2025** | modelo mucho mejor (786 árboles vs 362) |

### Y en este caso se puede tener lo mejor de los dos

Como las cancelaciones llegan hasta marzo de 2026, el modelo entrenado al
2026-01-03 **sí se puede calificar** contra enero — hay 3.965 fugas reales
registradas en ese mes. Resultado:

| | corte 2025-12-31 | **corte 2026-01-03** |
|---|---|---|
| filas de entrenamiento | 1.173.916 | **2.451.314** |
| árboles | 362 | **786** |
| AUC | 0,9028 | 0,9021 |
| PR-AUC | 0,4088 | **0,5671** |
| error de calibración top 5% | +2,1% | **−0,2%** |
| error top 1% | +7,6% | **−5,5%** (se queda corto, no sobrepromete) |

> **Corrección a lo dicho antes.** El script imprime *"métricas optimistas,
> label censurado"* al correr con corte en el mes en curso. Es una regla
> **conservadora y genérica**: asume que los desenlaces solo se conocen hasta la
> fecha de corte. **En estos datos esa advertencia es una falsa alarma**, porque
> el export de cancelaciones va hasta marzo de 2026. Los desenlaces de enero
> están completos y las métricas de arriba son legítimas — de hecho el modelo
> *subestima* en el tramo alto, lo contrario de ser optimista.

**Recomendación: reentrenar al corte 2026-01-03.** Es mejor en todo.

```bash
FORCE=1 CUTOFF_DATE=2026-01-03 bash scripts/run_all.sh
```

### ¿Hasta qué fecha se necesitan los datos, en general?

**Regla simple: para puntuar el mes M hacen falta los pagos hasta el último día
del mes M−1.**

| Para puntuar… | se necesitan los pagos hasta… | ¿están? |
|---|---|---|
| enero 2026 | 31-dic-2025 | ✅ sí |
| febrero 2026 | 31-ene-2026 | ❌ no |
| septiembre 2026 | 31-ago-2026 | ❌ no |

Los desenlaces (para volver a calificar el modelo) vienen del export de
cancelaciones, que hoy llega al 2026-03-16.

---

<a id="11"></a>
## 11. ¿Cómo corro la simulación de "quién se me va este mes"?

Eso es exactamente la **puntuación mensual**, y no requiere reentrenar: el modelo
entrenado con datos cerrados se aplica al listado de donantes activos del mes.

El archivo ya existe:
`results/motor2_donante/scoring_motor2_donante_2026_01_03.csv` — 127.206
donantes activos, ordenados de mayor a menor riesgo.

Es la etapa E/E2 de `run_monthly.sh`.

---

<a id="12"></a>
## 12. ¿Motor 2 acepta la fecha de ejecución? ¿cuál es el comando?

Sí. Motor 1 es un solo script (`--run-date`); Motor 2 son **dos pasos**:

```bash
# 1. Construir el retrato del mes (aquí, 1 de septiembre)
python3 scripts/02_features_motor2/01_motor2_dataset.py \
    --cutoff-date 2026-09-03 --months 1 \
    --out data/features/motor2_snapshot_2026_09_01.parquet

# 2. Puntuar — NO reentrena, carga el modelo guardado
python3 scripts/03_train/04b_score_motor2_donante.py \
    --score-date 2026-09-03 \
    --dataset data/features/motor2_snapshot_2026_09_01.parquet
```

O todo junto:

```bash
RUN_DATE=2026-09-03 PAYMENTS_YTD=raw/payments_2026_ytd.csv bash scripts/run_monthly.sh
```

*(Los nombres difieren por lo que significan: `--cutoff-date` es "hasta dónde
puedo mirar", `--score-date` es "qué día estoy corriendo esto". En Motor 1
coinciden y se llama `--run-date`.)*

### ⚠️ Pero hace falta que los datos estén al día

Con los pagos hasta el 31-dic-2025, pedir un retrato del 1-sep-2026 da esto:

| | enero (datos frescos) | septiembre (datos de hace 9 meses) |
|---|---|---|
| donantes | 127.206 | 121.420 |
| **P2 medio** | 0,0285 | **0,2716** ← 10x |
| días sin intento (mediana) | 31 | **274** |

El modelo ve que nadie ha pagado en 9 meses y concluye —correctamente, dado lo
que se le muestra— que la base entera está muerta. **Para la simulación de
septiembre hacen falta los pagos hasta agosto de 2026 en `raw/`.**

`run_monthly.sh` bloquea este caso (etapa C). Para la ejecución manual,
`01_motor2_dataset.py` avisa cuando el rezago pasa de 45 días.

---

<a id="13"></a>
## 13. `days_since_last_attempt_min` ¿contra qué fecha se calculó?

**Contra el snapshot (el día 1 del mes), no contra el score-date ni contra hoy.**
Verificado:

```
donante 003Qu00000Tsbw · último intento real 2025-11-28 · el CSV dice 34 días
  contra snapshot   (2026-01-01) →  34   <== ES ESTA
  contra score-date (2026-01-03) →  36
  contra hoy        (2026-09-03) → 279
```

**Y así debe ser.** El modelo aprendió con features medidas desde el primer día
del mes; si al puntuar se midieran desde otra fecha, las predicciones saldrían
sesgadas. Por eso conviene correr el proceso en los **primeros días del mes**:
entre más se demoren, más desactualizado queda ese "34 días" respecto al
calendario real.

---

<a id="14"></a>
## 14. ¿Cómo ve el equipo la salida?

**Respaldo:** `scripts/run_monthly.sh`

Archivos **CSV que se abren en Excel**, ordenados de mayor a menor riesgo. Nada
que instalar, ningún sistema al que entrar.

| Archivo | Filas | Qué es |
|---|---|---|
| `scoring_motor2_donante_<fecha>.csv` | 127.206 | **Lista de retención por persona** |
| `scoring_motor2_<fecha>.csv` | 128.428 | Lo mismo por compromiso individual |
| `timing_reintento_motor1.csv` | 177.883 | Qué día conviene reintentar cada cobro |
| `Tabla_Donantes_P1_P2_<corte>.csv` | 177.883 | Tabla maestra, 39 columnas |

Ejemplo real del 2026-01-03:

| contact_id | prob_churn_donante | n_pledges | streak_max | tenure_min | días sin intento | monto |
|---|---|---|---|---|---|---|
| 003Qu00000Tsbw… | 1,00 | 1 | 17 | 105 | 34 | 45.000 |
| 003Qu00000SrOY… | 0,96 | 1 | 24 | 126 | 30 | 40.000 |
| 003Qu00000TcXc… | 0,92 | 1 | 4 | 110 | 34 | 40.000 |

La primera columna útil es el riesgo. **Las demás están para que retención
entienda por qué** — no entran al cálculo, son contexto: el primero lleva 17
cobros rechazados seguidos y 34 días sin actividad, aportando 45.000/mes.

El archivo viene ordenado: **basta con tomar las primeras N filas.**

**Cuántos llamar** (medido sobre 2025):

| Si llaman a… | son | de esos se van | eso es | y alcanzan a |
|---|---|---|---|---|
| top 1% | 1.272 | 56% | 17x el promedio | 17% de las fugas |
| top 5% | 6.360 | 40% | 12x | **59% de las fugas** |
| top 10% | 12.720 | 25% | 7,6x | 76% de las fugas |
| top 20% | 25.441 | 14% | 4,2x | 83% de las fugas |

*"Llamando a 6.360 personas (el 5%), 4 de cada 10 llamadas son a alguien que
efectivamente se iba, y con eso se alcanza a más de la mitad de las fugas del
mes."* El punto de corte es decisión de negocio.

**El proceso mensual** es un comando y tarda minutos. Lo único manual es poner el
archivo de pagos del mes en `raw/`. **No reentrena** — eso es aparte, cada 6-12
meses.

---

<a id="15"></a>
## 15. ¿Valdría la pena incluir el monto de los pledges como feature?

**Sí, pero no el monto — la brecha.** Se probó con tres variantes:

| feature nueva | importancia | puesto |
|---|---|---|
| `monto_pactado_total` (suma comprometida) | 0,30% | 13 de 18 |
| `monto_pactado_max` | 0,51% | 12 de 18 |
| **`ratio_pagado_pactado`** | **40,46%** | **1 de 18** |

**El monto por sí solo no aporta casi nada.** Lo que aporta es **la brecha entre
lo que la persona se comprometió a dar y lo que realmente se le cobró.** Tiene
sentido: que alguien done 40.000 o 400.000 no dice si se va a ir; que se haya
comprometido a 40.000 y se le esté cobrando 0, sí.

| | AUC | PR-AUC | error top 5% |
|---|---|---|---|
| producción (15 features) | 0,8984 | 0,3769 | +7,3 pts |
| + monto (18 features) | 0,9009 | **0,4079** (+8%) | +9,4 pts ⚠️ |

**Tres advertencias antes de meterlo:**

1. **La calibración empeora** (+7,3 → +9,4 puntos). Habría que reajustar.
2. **Sola, la señal es débil.** Por tramos el lift va de 0,6x a 1,8x. El modelo la
   usa en *combinación* con otras, no como regla suelta.
3. **Riesgo temporal.** `npe03_amount_c` es el valor **de hoy** en el CRM, no el
   de la fecha del snapshot. Si alguien bajó su aporte en 2025, los snapshots de
   2024 llevarían el monto de 2025. **Se verificó lo peor** —que el CRM ponga el
   monto en 0 al cancelar— y **no pasa** (0,01% de ceros en cancelados vs 0,00%
   en activos), así que no hay fuga del desenlace. La inconsistencia temporal
   menor sigue ahí.

**Recomendación:** vale la pena, como cambio propio con su experimento — no
colarlo junto a lo demás.
