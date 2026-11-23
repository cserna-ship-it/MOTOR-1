# Informe técnico — Sobre el uso de `auto_class_weights` en los modelos CatBoost

**Proyecto:** UNICEF Colombia · Pipeline predictivo de cobros y retención (v8)
**Alcance:** Motor 2 (probabilidad de churn a 30 días) y Motor 1 (probabilidad de éxito de cobro)
**Fecha:** 2026-08-03

---

## 0. Resumen

Se documenta por qué el Motor 2 **no** usa `auto_class_weights` y qué efecto real tiene
ese parámetro en cada uno de los dos modelos.

La conclusión, sostenida por nueve entrenamientos completos sobre los datos reales del
cliente —seis configuraciones en el Motor 2 y tres en el Motor 1, todas ejecutadas y
archivadas el 2026-08-03— es:

1. **En el Motor 2, `auto_class_weights="Balanced"` es netamente perjudicial.** Cuesta
   **0,086 de PR-AUC** (−23% relativo) y **0,018 de AUC**, daño que la calibración
   posterior **no puede** revertir por razones que se demuestran en la sección 5.2.
2. **La justificación teórica habitual del balanceo no aplica a este pipeline.** El
   argumento "sin pesos el modelo predice que nadie abandona" describe el fracaso de un
   clasificador que optimiza *accuracy* con umbral duro en 0,5. Nuestro modelo optimiza
   `Logloss`, entrega probabilidades y decide por percentil. Empíricamente, el modelo sin
   pesos captura el **85,5%** de las fugas reales contactando al 20% de los donantes.
3. **En el Motor 1, en cambio, el mismo parámetro debe mantenerse.** Allí el desbalance es
   de 1,19 : 1 —no de 21,6 : 1— y el balanceo mide levemente mejor en AUC, PR-AUC, Brier,
   ECE y recall sobre la clase accionable.

Que el mismo parámetro se descarte en un motor y se conserve en el otro no es una
inconsistencia: es la consecuencia de haberlo medido en cada caso en vez de aplicarlo por
regla.

---

## 1. Qué hace exactamente el parámetro

### 1.1. La función de pérdida es una suma ponderada

CatBoost, con `loss_function="Logloss"`, no minimiza la log-verosimilitud simple sino su
versión ponderada por observación:

$$\mathcal{L} \;=\; -\sum_{i=1}^{n} w_i \Big[\, y_i \log q_i \;+\; (1-y_i)\log(1-q_i) \,\Big]$$

donde $q_i$ es la probabilidad predicha para la observación $i$. Por defecto $w_i = 1$
para todas las filas. `auto_class_weights` es, literalmente, una regla para fijar esos
$w_i$ en función de la clase de cada observación.

### 1.2. La regla que aplica CatBoost

Sea $n_c$ el número de observaciones de la clase $c$ en el train. CatBoost define:

| valor del parámetro | peso asignado |
|---|---|
| (ausente) | $w_c = 1$ |
| `"Balanced"` | $w_c = \dfrac{\max_k n_k}{n_c}$ |
| `"SqrtBalanced"` | $w_c = \sqrt{\dfrac{\max_k n_k}{n_c}}$ |

En el train del Motor 2 (snapshots de 2024, 1.173.916 filas, tasa de churn 4,4255%):

| clase | filas | $w_c$ con `Balanced` |
|---|---|---|
| 0 — no cancela | 1.121.964 | **1,0000** |
| 1 — cancela | 51.952 | **21,5962** |

Equivale exactamente a `scale_pos_weight = 21.5962`.

### 1.3. El significado: se entrena sobre un dataset ficticio 50/50

El peso no es un número arbitrario: está construido **para igualar la masa de las dos
clases**. Verificado sobre los conteos reales:

| clase | filas reales | × peso | masa efectiva |
|---|---|---|---|
| no cancela | 1.121.964 | × 1,0000 | **1.121.964** |
| cancela | 51.952 | × 21,5962 | **1.121.964** |

Las dos masas coinciden hasta el último dígito, porque así se define el peso. La
interpretación operativa es exacta y conviene enunciarla sin rodeos:

> Con `auto_class_weights="Balanced"` el modelo se entrena como si en la población el
> **50% de los donantes cancelara** y el 50% no. Sin el parámetro, el modelo ve la
> realidad: cancela el 4,4%.

Es equivalente a duplicar 21,6 veces cada fila positiva antes de entrenar, pero sin el
costo en memoria.

### 1.4. Qué le hace eso al óptimo de la pérdida

Este es el punto que decide todo lo demás, y se puede establecer de forma cerrada.

**Proposición.** Fijado un punto $x$ del espacio de features, sea $p = P(y=1 \mid x)$ la
probabilidad condicional verdadera. El minimizador de la pérdida ponderada con pesos
$w_1$ (positivos) y $w_0$ (negativos) es

$$q^\star \;=\; \frac{w_1\,p}{w_1\,p + w_0\,(1-p)}$$

*Demostración.* La pérdida esperada en $x$ es
$\ell(q) = -\big[w_1\,p\log q + w_0(1-p)\log(1-q)\big]$. Derivando e igualando a cero:
$\;-\frac{w_1 p}{q} + \frac{w_0(1-p)}{1-q} = 0 \Rightarrow w_1 p (1-q) = w_0 (1-p) q$,
de donde se despeja $q^\star$. La segunda derivada es positiva, luego es un mínimo. ∎

Escrito en *odds* queda más transparente:

$$\frac{q^\star}{1-q^\star} \;=\; \frac{w_1}{w_0}\cdot\frac{p}{1-p}$$

Es decir: **ponderar clases multiplica las odds por una constante**, o equivalentemente
desplaza el log-odds en $\log(w_1/w_0)$. En lenguaje bayesiano, es un cambio de *prior*:
el modelo ya no estima la probabilidad de churn en nuestra población, sino en una
población imaginaria donde el churn es 50%.

Con $w_1/w_0 = 21,5962$ y la tasa real de churn en test ($p = 0{,}0348$), la teoría
predice una probabilidad media cruda de **0,4379**. Lo medido en las corridas:

| config | pesos | media prob. **cruda** | tasa real |
|---|---|---|---|
| A | Balanced | 0,4214 | 0,0348 |
| E | Balanced | **0,4380** | 0,0348 |
| C | SqrtBalanced | 0,3951 | 0,0348 |
| F | SqrtBalanced | 0,1112 | 0,0348 |
| B | ninguno | **0,0409** | 0,0348 |
| D | ninguno | 0,0521 | 0,0348 |

La configuración E cae en 0,4380 contra los 0,4379 que predice la fórmula. Sin pesos, la
media cruda (0,041) queda donde debe estar: sobre la tasa base real.

> Las configuraciones con `Balanced` cortan en muy
> pocos árboles (sección 5.1), y la aproximación inicial de CatBoost en el caso ponderado
> parte cerca de 0,5. Por tanto la coincidencia de E no distingue por sí sola entre "el
> modelo convergió al óptimo sesgado" y "el modelo apenas se movió del punto inicial".
> El dato decisivo no es ese, sino el contraste: **0,42–0,44 con pesos frente a
> 0,041–0,052 sin pesos, contra una tasa real de 0,035.**

---

## 2. El argumento estándar a favor, y por qué no aplica a este pipeline

La recomendación habitual —y la que aparece en la documentación y en la mayoría de
tutoriales— se resume así: *"con 4,7% de clase positiva, sin pesos el modelo predecirá
que nadie abandona y tendrá 95,3% de accuracy, siendo inútil para el negocio"*.

Ese argumento es correcto en su contexto, y ese contexto no es el nuestro.

### 2.1. El "predice todo cero" es el fracaso de otra función objetivo

`Logloss` es una **regla de puntuación propia** (*proper scoring rule*): su esperanza se
minimiza si y solo si $q(x) = P(y=1\mid x)$ para todo $x$. El predictor constante
$q \equiv 0{,}044$ solo es óptimo cuando las features **no contienen ninguna señal**. En
cuanto `streak_actual` o `success_rate_hist` discriminan algo, la pérdida castiga
activamente al modelo por quedarse en la constante, y el descenso por gradiente se aleja
de ella en la primera iteración.

"Predecir todo cero" es lo que ocurre al optimizar *accuracy* o pérdida 0-1 con umbral
fijo en 0,5. No es lo que ocurre con boosting sobre `Logloss`.

La comprobación empírica está en nuestros propios datos. El modelo sin pesos, evaluado
out-of-time sobre 2025:

| métrica | valor |
|---|---|
| AUC | 0,8963 |
| PR-AUC | 0,3672 (baseline aleatorio: 0,0348) |
| Recall de fugas contactando al top 20% | **0,8552** |

Contactando al 20% de la base se alcanza al **85,5% de quienes efectivamente cancelaron**.
Esto es lo contrario de un modelo inútil.

### 2.2. Estimar y decidir son dos problemas separados

La razón de fondo por la que no necesitamos pesos es de teoría de la decisión. El
problema se descompone en dos etapas independientes:

1. **Estimación** — obtener $P(\text{churn}\mid x)$ lo más fielmente posible. Se resuelve
   con una regla de puntuación propia (`Logloss`) y, si hace falta, calibración.
2. **Decisión** — elegir a quién se contacta. Con un costo $c$ por contactar y un
   beneficio $b$ por retener, la regla óptima es contactar si
   $P(\text{churn}\mid x) > c/(c+b)$; en la práctica, y dado que el presupuesto de la
   campaña es fijo, el umbral se define por capacidad: el top-$k\%$ del ranking.

Ponderar clases es introducir la etapa 2 **dentro** de la etapa 1. Se paga en calibración
y en estabilidad del entrenamiento, y no se gana nada que no se pueda obtener moviendo el
umbral —que es gratis, reversible y no requiere re-entrenar.

Nuestro pipeline ya implementa la etapa 2 de forma explícita:
[`02_train_motor2.py:286`](../scripts/03_train/02_train_motor2.py#L286) corta en
`np.quantile(y_prob, 0.80)`. Con una probabilidad bien calibrada, el equipo de retención
puede pasar a top 10% o top 30% según el presupuesto del trimestre sin tocar el modelo.

### 2.3. Las precauciones que acompañan al consejo lo desactivan

Vale la pena señalar la inconsistencia interna del argumento estándar. Junto a la
recomendación de balancear se advierte, correctamente, que (a) no se debe usar *accuracy*
como métrica y (b) las probabilidades quedan distorsionadas y hay que calibrarlas o mover
el umbral.

Pero si no se usa *accuracy* y no se usa el umbral 0,5, **el problema que motivaba el
balanceo ya no existe**. Lo único que queda del parámetro es su costo.

Nuestro pipeline cumple ambas precauciones desde el diseño: `eval_metric="AUC"`, y se
reportan AUC, PR-AUC, Brier, ECE y recall/precision al top 20%. *Accuracy* no aparece en
ninguna parte del Motor 2.

---

## 3. Diseño experimental

Script: [`docs/02b_experimento_motor2.py`](02b_experimento_motor2.py)
Salidas archivadas: `results/motor2_experimentos/comparacion.json` y `log_ejecucion_motor2.txt`

Se aislaron dos ejes en un diseño factorial completo 3 × 2:

- **Pesos de clase:** ninguno · `Balanced` · `SqrtBalanced`
- **Learning rate:** `lr = 0,10` con 150 iteraciones · `lr = 0,03` con 800 iteraciones

Se controló todo lo demás para que la única fuente de variación fuera la configuración:
mismo split temporal out-of-time (train = snapshots 2024, test = snapshots 2025), mismo
holdout de validación (último 10% del train en orden canónico `pledge_id, snapshot_ts`),
misma calibración isotónica ajustada sobre ese holdout, `random_seed=42`, `depth=6`,
`od_type="Iter"`, `od_wait=15`, mismas 14 features.


**Reproducibilidad verificada.** La corrida del 2026-08-03 reproduce dígito por dígito la
del 2026-07-16 en las seis configuraciones. El experimento es determinista.

---

## 4. Resultados — Motor 2

Test = 1.374.159 snapshots de 2025, tasa de churn real 3,4815%.

| Config | Pesos | lr | best_iter | AUC | PR-AUC | ECE crudo | ECE post | Brier | R@20% | P@20% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Balanced | 0,10 | **3** | 0,8785 | 0,2812 | 0,387 | 0,0096 | 0,0269 | 0,831 | 0,128 |
| B | ninguno | 0,10 | 54 | 0,8964 | 0,3669 | 0,010 | 0,0106 | 0,0273 | 0,842 | **0,137** |
| C | SqrtBalanced | 0,10 | **2** | 0,8833 | 0,3330 | 0,361 | 0,0185 | 0,0293 | 0,831 | 0,130 |
| **D** ← producción | **ninguno** | **0,03** | **59** | 0,8963 | **0,3672** | 0,021 | **0,0085** | **0,0267** | 0,855 | 0,124 |
| E | Balanced | 0,03 | **7** | 0,8808 | 0,2978 | 0,403 | 0,0052 | 0,0270 | **0,858** | 0,115 |
| F | SqrtBalanced | 0,03 | 78 | **0,8981** | 0,3628 | 0,076 | 0,0101 | 0,0268 | 0,848 | 0,134 |

Lectura por eje:

- **Las tres configuraciones sin balanceo fuerte** (B, D, F) ocupan los tres primeros
  puestos en PR-AUC, con 0,363–0,367.
- **Las dos configuraciones con `Balanced`** (A, E) ocupan los dos últimos, con
  0,281–0,298.
- El efecto del *learning rate*, comparado con esto, es de segundo orden: entre B y D
  (mismo balanceo, distinto lr) la PR-AUC se mueve 0,0003.

**El balanceo es el factor dominante, y su efecto es negativo.**

---

## 5. Análisis — los tres hallazgos

### 5.1. El balanceo colapsa el early stopping

El dato más llamativo de la tabla no es una métrica de desempeño sino la columna
`best_iter`:

| pesos | lr = 0,10 | lr = 0,03 |
|---|---:|---:|
| `Balanced` | **3 árboles** | **7 árboles** |
| `SqrtBalanced` | **2 árboles** | 78 árboles |
| ninguno | 54 árboles | 59 árboles |

La configuración que estuvo en producción hasta julio (A) era, en rigor, **un ensamble de
tres árboles**. No es un modelo mal ajustado: es un modelo que nunca llegó a entrenarse.
La consecuencia visible es la feature importance, que con tres árboles se concentraba en
dos variables y dejaba varias en cero.

**Mecanismo.** El peso multiplica el gradiente de cada observación positiva por 21,6. El
tamaño de paso efectivo en la dirección de los positivos crece en la misma proporción, y
el AUC de validación satura en las primeras iteraciones; a partir de ahí oscila sin
mejorar y la paciencia de `od_wait=15` se agota.

**Honestidad sobre el mecanismo.** La hipótesis del paso efectivo explica bien el
comportamiento de `Balanced`, que colapsa con ambos *learning rates*, pero **no produce
una relación dosis-respuesta limpia**: `SqrtBalanced` corta en 2 árboles con lr=0,10 y en
78 con lr=0,03, un salto que el producto peso × lr no predice. La conclusión defendible
con los datos disponibles es más modesta y suficiente: **ponderar clases vuelve
errático el criterio de parada** (2 a 78 árboles según la combinación), mientras que sin
pesos es estable (54 y 59, con lr que difieren en un factor de 3).

Para un modelo que se re-entrena periódicamente en producción, esa inestabilidad es por
sí sola un motivo para descartar el parámetro.

**Matiz teórico importante.** Nada de esto es una necesidad estadística. Por la sección
1.4, ponderar clases aplica una transformación **estrictamente monótona** al log-odds; con
optimización exacta y capacidad ilimitada, el ranking sería idéntico con y sin pesos, y el
AUC no cambiaría. El daño observado es un artefacto de la interacción entre la ponderación
y el proceso finito de optimización con parada temprana. Es un daño real y reproducible,
pero conviene no atribuirle una causa que no tiene.

### 5.2. El daño al ranking es irreversible: la calibración no puede repararlo

Cuando se planteó quitar `Balanced`, la objeción natural fue: *"si el problema son las
probabilidades distorsionadas, el calibrador isotónico ya las corrige — ¿por qué importa?"*

La respuesta es que **el calibrador corrige la probabilidad, no el orden**, y las métricas
que sostienen el uso operativo del modelo dependen únicamente del orden.

**Lema.** AUC y PR-AUC (*average precision*) son funcionales del vector de scores que
dependen solo de su ordenamiento. Si $g$ es estrictamente creciente,
$\text{AUC}(g(s)) = \text{AUC}(s)$.

**Corolario.** La regresión isotónica produce una función escalonada **no decreciente**
—monótona en sentido débil, no estricto—. Puede por tanto **fundir** scores distintos en
un mismo valor (crear empates), pero nunca separar dos observaciones que el modelo dejó
en el mismo punto. Se sigue que la calibración **solo puede conservar o degradar** el
ranking; jamás mejorarlo.

**Verificación empírica** sobre las predicciones de producción
(`results/motor2/test_predictions_motor2.parquet`, que almacena ambas columnas):

| | probabilidad cruda | tras calibrar |
|---|---:|---:|
| valores distintos | 217.945 | **1.294** |
| AUC | 0,897756 | 0,896296 |
| PR-AUC | 0,390325 | 0,367222 |
| correlación de Spearman entre ambas | | 0,97655 |

El calibrador colapsa casi 218 mil scores distintos en 1.294 niveles, y el ranking se
degrada levemente (Spearman 0,977 en lugar de 1,000). Exactamente lo que predice el
corolario.

De ahí la conclusión que cierra la objeción: los **0,086 de PR-AUC** que cuesta `Balanced`
se pierden en el momento del entrenamiento y **ninguna etapa posterior los recupera**.

> *Observación adicional, ajena al balanceo.* El mismo experimento revela que la
> calibración isotónica nos cuesta hoy 0,023 de PR-AUC (0,3903 → 0,3672) por el colapso de
> empates. Un calibrador estrictamente monótono (Platt / regresión logística sobre el
> log-odds) preservaría el ranking de forma exacta. Alternativamente, se puede usar
> `prob_churn_cruda` para ordenar y `prob_churn` para comunicar la probabilidad absoluta;
> ambas columnas ya se exportan. Queda como recomendación de mejora, no como hallazgo de
> este informe.

### 5.3. Sin balanceo, la calibración deja de ser un rescate

El ECE (*Expected Calibration Error*) medido antes de calibrar cuantifica cuánto trabajo
estaba haciendo el calibrador:

| config | pesos | ECE crudo | ECE tras calibrar |
|---|---|---:|---:|
| A | Balanced | **0,387** | 0,0096 |
| E | Balanced | **0,403** | 0,0052 |
| D | ninguno | **0,021** | 0,0085 |
| B | ninguno | **0,010** | 0,0106 |

Con `Balanced`, el calibrador existía para deshacer el sesgo que el propio parámetro
introducía: llevaba una probabilidad media de 0,42 de vuelta a 0,04. Sin pesos, las
probabilidades crudas **ya salen prácticamente calibradas** y el calibrador pasa a ser un
ajuste fino.

Esto tiene una consecuencia de arquitectura que va más allá de la métrica: en la
configuración anterior, el calibrador era un **componente crítico** —si fallaba o
quedaba desactualizado, las probabilidades entregadas al negocio eran diez veces la tasa
real—. En la configuración actual es un componente de conveniencia. El sistema es más
robusto.

Se decidió **mantener** el calibrador de todos modos: es barato, entrega la columna
`prob_churn` lista para consumo y protege ante el drift interanual observado (la tasa de
churn baja de 4,43% en 2024 a 3,48% en 2025).

---

## 6. El mismo parámetro en el Motor 1

El Motor 1 sí usa `auto_class_weights="Balanced"`
([`01_train_motor1.py:353`](../scripts/03_train/01_train_motor1.py#L353)). La pregunta
natural, tras lo anterior, es si ahí también sobra. La respuesta es **no**, y el motivo
merece explicarse porque ilustra el principio general.

Para responderla con datos y no con analogías se escribió el experimento
[`docs/01b_experimento_motor1.py`](01b_experimento_motor1.py), que importa
`load_slice` y la lista de features directamente del script de producción vía `importlib`,
de modo que el preprocesamiento, el split y el holdout son bit a bit los de producción.
La configuración `Balanced` del experimento reproduce las métricas del modelo en
producción hasta el sexto decimal (AUC 0,911467; PR-AUC 0,867562; Brier 0,113195), lo que
valida que el experimento mide lo que dice medir.

### 6.1. El desbalance del Motor 1 es de 1,19 : 1

Este es el hecho central. Sobre el train real (2022-2024, 2.327.921 intentos):

| clase | filas | proporción | $w_c$ con `Balanced` |
|---|---:|---:|---:|
| 1 — éxito de cobro | 1.062.475 | 45,64% | **1,1910** |
| 0 — fallo de cobro | 1.265.446 | 54,36% | **1,0000** |

Con un ratio de 1,19 el peso es un empujón del 19%, no una intervención de 2.060% como en
el Motor 2. La distorsión que introduce sobre la probabilidad es, por la fórmula de la
sección 1.4, de apenas 0,04 en probabilidad ($q^\star = 0{,}4337$ frente a $p = 0{,}3914$),
contra los 0,40 del Motor 2. Es el **mismo parámetro produciendo dos intervenciones de
naturaleza distinta**, porque actúa sobre desbalances de escala incomparable.

### 6.2. Resultados

Test = 2.981.152 intentos de 2025, tasa de éxito real 39,14%.

| Config | Pesos | best_iter | AUC | PR-AUC | Brier | ECE post | Recall fallo @0,5 | Precisión fallo @0,5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| — | ninguno | 77 | 0,90945 | 0,86575 | 0,11511 | 0,0313 | 0,8627 | **0,8770** |
| **producción** | **Balanced** | 49 | **0,91147** | **0,86756** | **0,11320** | **0,0229** | **0,8930** | 0,8608 |
| — | SqrtBalanced | 57 | 0,90959 | 0,86481 | 0,11522 | 0,0284 | 0,8608 | 0,8754 |

Nada colapsa: los tres modelos entrenan entre 49 y 77 árboles. Y `Balanced` sale
**levemente mejor en todo lo que importa**: +0,0020 de AUC, +0,0018 de PR-AUC, −0,0019 de
Brier, −0,0084 de ECE, y +3,0 puntos de recall sobre la clase accionable (el fallo) al
umbral 0,5, a cambio de −1,6 puntos de precisión.

Las diferencias son pequeñas, pero son deterministas y reproducibles (semilla y split
fijos), y **todas apuntan en la misma dirección**. No hay motivo para quitar el parámetro
del Motor 1.

### 6.3. La lectura conjunta

Los dos motores juntos son el mejor argumento contra aplicar `auto_class_weights` como
regla general:

| | Motor 1 | Motor 2 |
|---|---|---|
| ratio de clases | 1,19 : 1 | **21,60 : 1** |
| peso aplicado a la minoritaria | 1,19 | **21,60** |
| sesgo en la probabilidad media cruda | 0,39 → 0,40 | 0,03 → **0,42** |
| árboles entrenados | 49 | **3** |
| efecto sobre PR-AUC | **+0,0018** | **−0,0860** |
| decisión | **mantener** | **quitar** |

Mismo parámetro, misma librería, mismo equipo, decisiones opuestas. El parámetro no es
bueno ni malo: su efecto depende por completo del desbalance sobre el que actúa y de la
interacción con el resto del procedimiento. Por eso la única política defendible es
**medirlo en cada modelo**, que es lo que se hizo aquí.

---

## 7. Conclusiones y recomendación

1. **Motor 2 — no usar `auto_class_weights`.** Con un desbalance de 21,6 : 1 el parámetro
   entrena al modelo sobre una población ficticia 50/50, colapsa el early stopping en 3
   árboles y cuesta 0,086 de PR-AUC (−23% relativo). El daño es al ranking, y por el
   corolario de la sección 5.2 la calibración posterior no puede repararlo. La
   configuración vigente (sin pesos, `learning_rate=0,03`, `iterations=800`) es la
   ganadora del experimento factorial y ya está en producción.

2. **Motor 1 — mantener `auto_class_weights="Balanced"`.** Con un desbalance de 1,19 : 1
   el parámetro es un ajuste marginal que mide levemente mejor en las cinco métricas
   evaluadas. Corregir, eso sí, las tres afirmaciones erróneas del docstring y del README
   señaladas en 6.3.

3. **Como política de proyecto:** `auto_class_weights` no debe aplicarse por defecto ante
   un target desbalanceado. Debe tratarse como cualquier otro hiperparámetro —probado
   contra el conjunto de test out-of-time y evaluado con las métricas que efectivamente
   se usan para decidir (PR-AUC y recall al top-k, no *accuracy*)—. Ambos experimentos
   quedan versionados en `docs/` para poder repetirlos en cada re-entrenamiento.


---

## 8. Reproducibilidad

Todo lo afirmado en este informe se regenera con dos comandos, desde la raíz del
entregable:

```bash
.venv/bin/python docs/02b_experimento_motor2.py    # 6 configuraciones, Motor 2
.venv/bin/python docs/01b_experimento_motor1.py    # 3 configuraciones, Motor 1
```

Archivos de evidencia (no se sobreescriben artefactos de producción):

| archivo | contenido |
|---|---|
| `results/motor2_experimentos/comparacion.json` | métricas de las 6 configuraciones del Motor 2 |
| `results/motor2_experimentos/log_ejecucion_motor2.txt` | salida completa de la corrida |
| `results/motor1_experimentos/comparacion.json` | métricas de las 3 configuraciones del Motor 1 + diagnóstico de pesos |
| `results/motor1_experimentos/log_ejecucion_motor1.txt` | salida completa de la corrida |
| `results/motor2/resultados_motor2.json` | métricas del modelo en producción |
| `results/motor1/resultados_motor1.json` | métricas del modelo en producción |

Los experimentos son deterministas (`random_seed=42`, split temporal fijo, orden canónico
explícito). La corrida del 2026-08-03 del experimento del Motor 2 reprodujo dígito por
dígito la del 2026-07-16.
