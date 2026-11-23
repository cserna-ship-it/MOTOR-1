#!/usr/bin/env python3
"""
config.py — Parámetros globales del pipeline UNICEF Colombia

Centraliza la fecha de corte (CUTOFF_DATE) para que todos los scripts
la lean del mismo lugar. Acepta tres fuentes, en orden de prioridad:

  1. Argumento CLI:          --cutoff-date 2026-03-31
  2. Variable de entorno:    CUTOFF_DATE=2026-03-31
  3. Default hard-coded:     2025-12-31  (comportamiento original v8)

Uso desde cualquier script del pipeline:

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/
    from config import get_cutoff_date

    CUTOFF = get_cutoff_date()          # pd.Timestamp
    CUTOFF_YEAR  = CUTOFF.year          # 2025 o 2026 ...
    CUTOFF_SLUG  = CUTOFF.strftime("%Y_%m_%d")  # "2025_12_31"
    CUTOFF_MONTH = CUTOFF.replace(day=1)         # primer día del mes de corte

Ejecutar el pipeline con fecha distinta:

    # Por variable de entorno (recomendado para run_all.sh):
    CUTOFF_DATE=2026-03-31 bash scripts/run_all.sh

    # O directamente al script:
    python3 scripts/03_train/01_train_motor1.py --cutoff-date 2026-03-31
"""

import os
import argparse
import pandas as pd

_DEFAULT_CUTOFF = "2025-12-31"


def get_cutoff_date() -> pd.Timestamp:
    """
    Devuelve la fecha de corte como pd.Timestamp.

    Prioridad: --cutoff-date CLI > env CUTOFF_DATE > default 2025-12-31.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=None,
        help="Fecha de corte YYYY-MM-DD (ej: 2026-03-31)",
    )
    args, _ = parser.parse_known_args()

    raw = args.cutoff_date or os.environ.get("CUTOFF_DATE") or _DEFAULT_CUTOFF

    try:
        cutoff = pd.Timestamp(raw)
    except Exception:
        raise ValueError(
            f"CUTOFF_DATE inválida: '{raw}'. Usa formato YYYY-MM-DD (ej: 2026-03-31)."
        )

    print(f"  [config] CUTOFF_DATE = {cutoff.date()}  "
          f"(fuente: {'CLI' if args.cutoff_date else 'ENV' if os.environ.get('CUTOFF_DATE') else 'default'})")
    return cutoff


# =============================================================================
# Partición de validación para early stopping y calibración
# =============================================================================
# CAMBIO v9 — antes: `train.iloc[-10%:]`, o sea el último 10% del train ordenado
# por (pledge_id | contact_id, snapshot_ts).
#
# Ese corte "por posición" resultaba ser un corte TEMPORAL encubierto: los IDs de
# Salesforce se asignan en orden de creación, así que ordenar por ID ≈ ordenar
# por antigüedad del donante, y los donantes nuevos se van más. El validation set
# quedaba con 5,90% de churn contra 4,26% del resto del train y 3,48% del test.
#
# Dos consecuencias, las dos malas:
#   1. El early stopping (que mira AUC en ese val) cortaba el entrenamiento
#      demasiado pronto: 59 árboles en pledge, 20 en donante.
#   2. El calibrador isotónico se ajustaba contra esa muestra pesimista y
#      heredaba el sesgo, sobreestimando justo en el tramo alto — el que usa
#      retención. Prometía 90% de abandono donde ocurría 56%.
#
# Ver docs/06_experimento_calibracion_motor2.py y docs/07_Informe_calibracion.md.

MESES_VALIDACION = 2  # meses finales del train reservados para validación


def split_validacion_temporal(train, col_snapshot="snapshot_ts",
                              col_label="label_churn_30d",
                              meses=MESES_VALIDACION, verbose=True):
    """
    Parte `train` en (tr_idx, val_idx) usando los últimos `meses` snapshots como
    validación, en vez del último 10% por posición.

    Devuelve dos Index de pandas, listos para `train.loc[...]`.

    Imprime la tasa del evento en cada parte. NO abortar por lo que salga ahí,
    pero SÍ mirarlo: el calibrador solo puede ser tan bueno como representativa
    sea esta muestra. Si la tasa del val se aleja mucho de la del último mes del
    train, el número que el modelo le entrega a negocio se va a leer sesgado.
    """
    meses_disponibles = sorted(train[col_snapshot].unique())
    if len(meses_disponibles) <= meses:
        raise SystemExit(
            f"\n[ERROR] El train tiene {len(meses_disponibles)} snapshot(s) y se piden "
            f"{meses} para validación — no queda nada con qué entrenar.\n"
            f"        Baja MESES_VALIDACION en scripts/config.py o amplía la ventana.\n")

    corte = meses_disponibles[-meses]
    es_val = train[col_snapshot] >= corte
    tr_idx, val_idx = train.index[~es_val], train.index[es_val]

    if verbose:
        tasa_tr = train.loc[tr_idx, col_label].mean()
        tasa_val = train.loc[val_idx, col_label].mean()
        ultimo_mes = train[train[col_snapshot] == meses_disponibles[-1]][col_label].mean()
        print(f"  [split] validación = últimos {meses} mes(es) del train "
              f"(desde {pd.Timestamp(corte):%Y-%m}): {len(val_idx):,} filas "
              f"({100 * len(val_idx) / len(train):.0f}%)")
        print(f"  [split] tasa del evento — train {tasa_tr:.4f} | val {tasa_val:.4f} "
              f"| último mes {ultimo_mes:.4f}")
        if ultimo_mes > 0 and abs(tasa_val - ultimo_mes) / ultimo_mes > 0.25:
            print(f"  [AVISO] la tasa del val se aleja >25% de la del último mes. "
                  f"El calibrador puede quedar sesgado: revisar MESES_VALIDACION "
                  f"(ver docs/07_Informe_calibracion.md).")
    return tr_idx, val_idx


def tabla_confiabilidad(prob, y, tramos=(0.01, 0.05, 0.10, 0.20)):
    """
    ¿El número que el modelo le entrega a negocio se puede leer literal?

    Para cada tramo superior (el top 1%, 5%, ... — la gente a la que retención
    va a llamar) compara lo que el modelo PROMETE contra lo que OCURRE.

    Es la métrica que le importa al negocio y no la captura ni el AUC (que solo
    mide el orden) ni el ECE global (que se lo come el 95% de casos de bajo
    riesgo). Se guarda en el JSON de resultados de cada entrenamiento: si un
    reentrenamiento futuro rompe la calibración, aparece acá.

    Devuelve lista de dicts, uno por tramo.
    """
    import numpy as np
    prob, y = np.asarray(prob, dtype=float), np.asarray(y, dtype=float)
    orden = np.argsort(-prob)
    filas = []
    for pct in tramos:
        k = max(int(len(prob) * pct), 1)
        idx = orden[:k]
        promete, ocurre = float(prob[idx].mean()), float(y[idx].mean())
        filas.append({
            "tramo": f"top_{int(pct*100)}pct",
            "n": int(k),
            "promete": round(promete, 4),
            "ocurre": round(ocurre, 4),
            "error": round(promete - ocurre, 4),
            "lift": round(ocurre / y.mean(), 2) if y.mean() > 0 else None,
        })
    return filas


def imprimir_confiabilidad(filas, tasa_base):
    print(f"  --- ¿el número se puede leer literal? (tasa base {tasa_base:.2%}) ---")
    print(f"  {'tramo':>10s} {'promete':>9s} {'ocurre':>9s} {'error':>9s} {'lift':>7s}")
    for f in filas:
        print(f"  {f['tramo']:>10s} {f['promete']:>8.1%} {f['ocurre']:>9.1%} "
              f"{f['error']:>+9.1%} {str(f['lift'])+'x':>7s}")
