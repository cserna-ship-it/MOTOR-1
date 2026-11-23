# Comparación Motor 2 — v2 actual vs alternativas de configuración

**Fecha:** 2026-07-16
**Script:** `docs/02b_experimento_motor2.py`
**Salida:** `results/motor2_experimentos/comparacion.json` (no toca producción)

Se compararon 6 configuraciones sobre el **mismo** split temporal, mismo val holdout,
misma calibración isotónica y mismo `random_seed=42`. Se aislaron dos ejes:

- **Pesos de clase:** `Balanced` · sin pesos (`None`) · `SqrtBalanced`
- **Learning rate:** `lr=0.10 / iters=150` · `lr=0.03 / iters=800`

Ratio real de clases en train: **21.6 negativos por cada positivo** (churn ~4.4%).

---

## Tabla de resultados (test = 2025)

| Config | Pesos | lr | **best_iter** | **AUC** | **PR-AUC** | ECE post-calib | Brier | R@20% | P@20% |
|---|---|---|---|---|---|---|---|---|---|
| **A — v2 actual** | Balanced | 0.10 | **3** | 0.8785 | 0.2812 | 0.0096 | 0.0269 | 0.831 | 0.128 |
| B — sin pesos | None | 0.10 | 54 | 0.8964 | 0.3669 | 0.0106 | 0.0273 | 0.842 | **0.137** |
| C — sqrt | SqrtBalanced | 0.10 | 2 | 0.8833 | 0.3330 | 0.0185 | 0.0293 | 0.831 | 0.130 |
| **D — sin pesos, lr baja** | None | 0.03 | 59 | 0.8963 | **0.3672** | **0.0085** | **0.0267** | 0.855 | 0.124 |
| E — balanced, lr baja | Balanced | 0.03 | 7 | 0.8808 | 0.2978 | 0.0052 | 0.0270 | **0.858** | 0.115 |
| **F — sqrt, lr baja** | SqrtBalanced | 0.03 | 78 | **0.8981** | 0.3628 | 0.0101 | 0.0268 | 0.848 | 0.134 |

*(en negrita el mejor valor de cada columna; A es la config de producción actual)*

---

## Respuesta directa a la pregunta: ¿quitar `Balanced`? ¿lo maneja todo el calibrador?

**Sí, conviene quitar `Balanced`. Y no, el calibrador no "lo maneja todo": corrige la
probabilidad, pero no puede corregir el daño que `Balanced` hace al ordenamiento y al
entrenamiento.** La evidencia:

1. **`Balanced` provocaba el early stopping en 4 árboles.**
   Con pesos → `best_iter = 3` (A) / `7` (E). Sin pesos → `best_iter = 54` (B) / `59` (D).
   El balanceo hacía que el modelo saturara la AUC del val en 3 árboles y el detector de
   overfitting lo cortara. Sin pesos el modelo entrena con normalidad.

2. **Quitar `Balanced` sube AUC y sobre todo PR-AUC.**
   AUC: 0.878 → 0.896 (+0.018). **PR-AUC: 0.281 → 0.367 (+0.086, ~30% relativo).**
   La PR-AUC es la métrica que más importa aquí (clase positiva rara), y es la que más mejora.
   El calibrador **no puede** recuperar esto: la calibración es monótona, no cambia el orden.

3. **El calibrador estaba, en gran parte, solo limpiando el desastre de `Balanced`.**
   ECE *crudo* (antes de calibrar): con `Balanced` = **0.39**; sin pesos = **0.010**.
   Es decir, sin `Balanced` las probabilidades crudas **ya salen casi calibradas** (media
   cruda ≈ tasa real). El calibrador pasa de ser un rescate a ser un ajuste fino casi
   imperceptible (ECE final 0.010 → 0.010).

**Conclusión conceptual:** `Balanced` en este caso era contraproducente. Para AUC/ranking los
pesos son casi irrelevantes en teoría, pero en la práctica aquí *rompían* el entrenamiento
(early stop prematuro). El calibrador arreglaba la probabilidad final pero no el ranking, así
que el neto de `Balanced` era negativo.

> Nota: `Balanced` equivale a `scale_pos_weight ≈ 21.6`. No es que "esté mal balanceado";
> es que con logloss + calibración posterior **no se necesita** balancear, porque el
> calibrador ya lleva las probabilidades a la tasa real sin sacrificar el ranking.

---

## Recomendación

**Config D — sin pesos, `learning_rate=0.03`, `iterations=800`** como opción principal:

- Mejor PR-AUC (0.3672) y mejor ECE (0.0085) del conjunto.
- Recall@20% = 0.855 (vs 0.831 actual): con el mismo 20% de donantes contactados se capturan
  **2.4 puntos más** de churners reales.
- lr baja + más iteraciones = modelo más robusto y menos dependiente de un corte temprano.

**Config F — `SqrtBalanced`, lr=0.03** es una alternativa casi empatada (AUC más alta, 0.8981;
P@20% más alta entre las lr bajas). Si se prefiere mantener algún balanceo suave, es la mejor.

Cualquiera de B / D / F es netamente superior a la config A de producción. La decisión entre
ellas es de segundo orden; lo importante es **quitar `Balanced`**.

---

## Otras sugerencias (más allá del balanceo)

1. **Bajar `learning_rate` a 0.03 y subir `iterations`** (ya incluido arriba). Aporta un extra
   marginal sobre lr=0.1 y da un modelo más estable.

2. **Calibrar en un set separado del early stopping.** Hoy el calibrador se ajusta sobre el
   *mismo* val holdout que se usó para elegir el `best_iteration` (`use_best_model=True`), lo
   que introduce un leve optimismo. Idealmente: dividir train en train / val (early stopping) /
   calib (calibración) disjuntos. Impacto esperado: pequeño (el ECE test ya es sano), pero es
   más correcto metodológicamente. *No se probó en este experimento para no mezclar variables.*

3. **La calibración es casi opcional sin `Balanced`.** Con ECE crudo de 0.010 el calibrador
   apenas mueve la aguja. Recomiendo **mantenerlo igual** (es barato, da la columna
   `prob_churn` lista para producción y protege ante drift), pero ya no es un componente
   crítico como lo era con `Balanced`.

4. **Features con importancia 0.** Al entrenar más árboles (54 vs 4) algunas features antes
   ignoradas podrían empezar a aportar. Vale la pena re-revisar la importancia con la config
   nueva antes de decidir podar variables.

5. **Drift interanual 2024→2025.** La tasa de churn baja de 4.4% a 3.5%. El calibrador queda
   anclado a 2024. Si en producción se observa este patrón, considerar re-calibrar
   periódicamente con datos recientes.

---

## Estado: APLICADO ✅ (2026-07-16)

La config ganadora **D** (sin pesos, `learning_rate=0.03`, `iterations=800`) se portó al script
de producción `02_train_motor2.py` y se regeneraron los artefactos de `results/motor2/`.
Verificación v2 (backup) vs v3 (producción actual):

| métrica | v2 (Balanced) | v3 (sin pesos) | Δ |
|---|---|---|---|
| AUC | 0.8785 | **0.8963** | +0.0178 |
| PR-AUC | 0.2812 | **0.3672** | +0.0860 |
| Brier calibrado | 0.0269 | **0.0267** | −0.0002 |
| ECE test (post) | 0.0096 | **0.0085** | −0.0011 |
| Recall @top20% | 0.8312 | **0.8552** | +0.0241 |
| Precision @top20% | **0.1280** | 0.1237 | −0.0043 |
| best_iteration | 3 | **59** | modelo sano |

Único retroceso: Precision@20% (−0.0043), intercambio menor frente a +0.0241 de recall al mismo
corte. Todo lo demás mejora. La feature importance de v3 se reparte más (streak_actual,
success_rate_hist, tenure_days, cum_fallos, cum_exitos) al usar 59 árboles en vez de 3.

**Tabla maestra Motor 3 regenerada** (`data/entregable/Tabla_Donantes_P1_P2_2025_12_31.{csv,parquet}`,
cutoff 2025-12-31) con las predicciones v3. La distribución de P2 pasa de un patrón con
"escalones" (artefacto del calibrador v2) a un gradiente de riesgo usable para priorización:

| columna P2 | v2: p95 | v3: p95 | v2: >0.6 | v3: >0.6 | v2: >0.8 | v3: >0.8 |
|---|---|---|---|---|---|---|
| p2_ultimo | 0.2424 | **0.6944** | 7.614 | **18.001** | 7.613 | 5.719 |
| p2_max | 0.2424 | **0.6944** | 7.936 | **21.064** | 7.936 | 6.691 |
| p2_dec_2025 | 0.2424 | **0.1580** | 697 | 3.034 | 696 | 816 |

En v2, `>0.6` y `>0.8` daban casi el mismo conteo (nada entre 0.6 y 0.8): el calibrador
saltaba de 0.24 a 0.92 sin valores intermedios. En v3 hay un gradiente real, con 18.001
pledges sobre 0.6 de los cuales 5.719 superan 0.8 — mucho más accionable para segmentar
campañas de retención. La media de P2 dic-2025 (0.041) sigue alineada con la tasa base (~4%).
