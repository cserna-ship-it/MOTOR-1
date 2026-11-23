# Informe: por qué "90%" no significaba 90%, y cómo se arregló

**Motor 2 · corte 2025-12-31 · septiembre de 2026**

---

## 1. El problema que reportó negocio

El Motor 2 ordenaba muy bien a los donantes por riesgo (AUC 0,89), pero el
**número** que entregaba no se podía leer literal:

> el modelo dice **90%** → se van de verdad **56%**

Retención no puede planear con eso. Necesita poder decir *"de cada 100 donantes
así, se van N"* y que N sea cierto.

---

## 2. Diagnóstico: dos causas, ninguna evidente

### Causa 1 — `boost_from_average=False`

Es el **default de CatBoost para `Logloss`** (no lo puso nadie a mano; para RMSE
el default es el contrario). El modelo arranca prediciendo 0,5 —una moneda al
aire— en vez de la tasa base (~3,5%), y va bajando árbol por árbol.

El problema es que el early stopping vigila **AUC**, o sea el *orden*. Y el
orden deja de mejorar mucho antes de que el *número* llegue a su valor correcto.
Resultado: el entrenamiento se cortaba a mitad de camino.

| | árboles | media de la probabilidad cruda | tasa real |
|---|---|---|---|
| pledge | 59 | 0,052 | 0,035 |
| donante | 20 | **0,139** | 0,033 |

### Causa 2 — el validation set no era representativo

Se tomaba como *el último 10% del train ordenado por `pledge_id` / `contact_id`*.

Los IDs de Salesforce se asignan en orden de creación, así que **ordenar por ID
es casi ordenar por antigüedad del donante**. Ese corte "por posición" era en
realidad un corte temporal encubierto que se quedaba con los donantes más
nuevos — los que más se van.

| grupo | tasa de churn |
|---|---|
| validation (posicional) | **5,90%** |
| resto del train | 4,26% |
| test 2025 (la realidad) | 3,48% |

La tasa de churn por decil de ese ordenamiento va de **0,0174 a 0,0623** (3,6x).

Eso golpeaba dos veces: el early stopping cortaba temprano contra una muestra
pesimista, y el **calibrador isotónico se ajustaba contra esa misma muestra** y
heredaba el sesgo — justo en el tramo alto, que es el que usa retención.

---

## 3. El experimento

`docs/06_experimento_calibracion_motor2.py` compara cuatro configuraciones. La
métrica que decide **no es el AUC** (el orden ya estaba bien): es el error de
calibración dentro del top 1% / 5% / 10%, la población a la que se llama.

| config | | error medio tramo alto (pledge) | error medio (donante) |
|---|---|---|---|
| A | actual: `bfa=False`, split posicional | 0,162 | 0,229 |
| B | `bfa=True`, split posicional | **0,185** | **0,248** |
| C | `bfa=False`, split temporal | 0,059 | 0,069 |
| D | `bfa=True`, split temporal | **0,036** | 0,074 |

**Hallazgo clave: la causa dominante era el validation set, no
`boost_from_average`.** De hecho `boost_from_average=True` **por sí solo empeora
las cosas** (config B): con el val sesgado el modelo para en 0 árboles (pledge) y
3 (donante), y la PR-AUC se desploma de 0,367 a 0,296. Las dos correcciones solo
funcionan **juntas**.

Se eligió **D** para los dos modelos: gana claro en pledge, empata con C en
donante (0,074 vs 0,069, diferencia de ruido) pero le gana en PR-AUC
(0,377 vs 0,358), y tener una sola configuración para ambos es preferible.

### Advertencia: el tamaño de la ventana importa

Se probó con 1, 2 y 3 meses de validación:

| meses de val | tasa del val | error tramo alto (pledge) |
|---|---|---|
| 1 | 0,0408 | 0,068 |
| **2** | **0,0387** | **0,036** |
| 3 | 0,0439 | 0,155 ← casi tan malo como el original |

Con 3 meses entra octubre-2024 (5,43% de churn, un pico) y contamina la muestra.
**El calibrador solo puede ser tan bueno como representativa sea la ventana
contra la que se ajusta.** Por eso `split_validacion_temporal()` imprime la tasa
del val contra la del último mes y **avisa si se separan más de 25%**. Si se
mueve la fecha de corte, hay que mirar ese aviso.

---

## 4. Qué se cambió

`scripts/config.py`
- `MESES_VALIDACION = 2` y `split_validacion_temporal()` — reemplaza el corte
  posicional. Imprime las tasas y avisa si el val no es representativo.
- `tabla_confiabilidad()` / `imprimir_confiabilidad()` — guardarraíl permanente.

`scripts/03_train/02_train_motor2.py` y `02b_train_motor2_donante.py`
- Usan la partición temporal.
- `boost_from_average=True`.
- Cada entrenamiento imprime y guarda en el JSON
  (`catboost.confiabilidad_tramo_alto`) la comparación promete-vs-ocurre. Si un
  reentrenamiento futuro rompe la calibración, queda registrado.

---

## 5. Resultados

|  | pledge antes | pledge después | donante antes | donante después |
|---|---|---|---|---|
| AUC | 0,8963 | **0,9028** | 0,8932 | **0,8984** |
| PR-AUC | 0,3672 | **0,4088** *(+11%)* | 0,3698 | **0,3769** |
| Brier calibrado | 0,0267 | **0,0241** | 0,0266 | **0,0240** |
| ECE calibrado | 0,0085 | **0,0023** | 0,0125 | **0,0053** |
| árboles | 59 | **362** | 20 | **362** |

**Beneficio inesperado:** el val sesgado no solo descalibraba, **estaba cortando
el entrenamiento prematuramente**. Al arreglarlo el modelo entrena 6-18x más
árboles y la PR-AUC del pledge sube 11%. Era pérdida de poder predictivo real,
no solo un problema de presentación.

### Lo que le importa a negocio

**Pledge**

| tramo | antes: dice → ocurre | después: dice → ocurre |
|---|---|---|
| top 1% | 83% → 55% *(+28 pts)* | 67% → 60% *(+8 pts)* |
| top 5% | 54% → 40% *(+13 pts)* | 46% → 44% *(+2 pts)* |
| top 10% | 34% → 27% *(+7 pts)* | 28% → 27% *(+1 pt)* |
| top 20% | — | 15% → 15% *(+0,4 pts)* |

**Donante**

| tramo | antes: dice → ocurre | después: dice → ocurre |
|---|---|---|
| top 1% | 90% → 56% *(+34 pts)* | 67% → 56% *(+11 pts)* |
| top 5% | 62% → 39% *(+23 pts)* | 47% → 40% *(+7 pts)* |
| top 10% | 36% → 25% *(+11 pts)* | 29% → 25% *(+4 pts)* |

---

## 6. Cómo debe leerlo negocio ahora

**Del top 5% hacia abajo el número se puede leer literal.** Si `prob_churn` dice
0,44 significa que de cada 100 donantes así, ~44 se van en los próximos 30 días.

**En el extremo (top 1%, tramo 0,8-1,0) todavía sobreestima** ~7-25 puntos. Es
una limitación del calibrador isotónico en el borde: hay muy pocos casos ahí y
`out_of_bounds="clip"` los fija en 1,0. En ese tramo léase como **"riesgo muy
alto"**, no como un porcentaje exacto. Sigue siendo el grupo correcto para
llamar primero: el lift es 17x.

Tabla operativa (medida sobre el test 2025, modelo de donante):

| segmento | donantes | de estos se van | lift | % de todas las fugas que captura |
|---|---|---|---|---|
| top 1% | 13.587 | 56% | 17x | 17% |
| top 5% | 67.939 | 40% | 12x | 59% |
| top 10% | 135.879 | 25% | 7,6x | 76% |
| top 20% | 271.758 | 14% | 4,2x | 83% |

---

## 7. Motor 1: mismo defecto, impacto menor

Motor 1 usa el mismo split posicional. Pero su target (`is_success`) tiene tasa
base 39%, no 3,5%, así que el sesgo del val pesa mucho menos. Su calibración
medida:

| tramo de prob. | dice | ocurre | error |
|---|---|---|---|
| 0,0-0,2 | 10,6% | 8,9% | +1,7 |
| 0,2-0,4 | 28,0% | 32,4% | −4,4 |
| 0,4-0,6 | 49,2% | 51,8% | −2,6 |
| 0,6-0,8 | 74,9% | 74,4% | +0,5 |
| 0,8-1,0 | 88,7% | 92,3% | −3,6 |

Errores de ±4 puntos y **sin dirección sistemática** — no es el mismo problema.
El único síntoma es que su calibrador empeora ligeramente el ECE de test
(0,0180 → 0,0229), señal de que el val tampoco lo representa bien.

**No se cambió.** Vale la pena probar la misma corrección en el próximo ciclo,
pero no es urgente y Motor 1 está afinado con otra configuración
(`auto_class_weights="Balanced"`) que habría que revalidar.

---

## 8. Pendiente

- Reproducir el experimento cuando cambie la fecha de corte, y mirar el aviso de
  `split_validacion_temporal()`: la ventana de 2 meses está afinada contra este
  corte, no es universal.
- Evaluar la misma corrección en Motor 1.
- Para el extremo del ranking, considerar un calibrador que extrapole mejor en el
  borde (p. ej. Platt scaling sobre el último decil) si negocio necesita leer el
  top 1% de forma literal.
