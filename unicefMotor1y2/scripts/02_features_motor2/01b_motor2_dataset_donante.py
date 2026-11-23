#!/usr/bin/env python3
"""
01b_motor2_dataset_donante.py — Motor 2 a nivel DONANTE

POR QUÉ EXISTE ESTE SCRIPT
---------------------------
El Motor 2 pledge-level (01_motor2_dataset.py + 02_train_motor2.py) predice la
cancelación de un PLEDGE. El negocio necesita la cancelación del DONANTE: si un
donante tiene dos pledges y cancela uno pero mantiene el otro, el donante NO
abandonó — sigue dando. El label a nivel pledge no distingue esto, y por lo
tanto tampoco lo distinguen las decisiones de intervención que se toman con él.

REGLA DE NEGOCIO
----------------
Un donante "abandona" en la ventana (t, t+30d] si y solo si TODOS los pledges
que tenía activos en el snapshot t cancelan dentro de esa ventana:

    label_donante(d, t) = 1  ⟺  label_churn_30d(p, t) = 1  ∀ p ∈ P(d, t)

Como label_churn_30d ya es 0/1 a nivel pledge, "para todo p vale 1" es
exactamente el MÍNIMO del grupo (contact_id, snapshot_ts).

ESTRATEGIA: AGREGAR, NO RECONSTRUIR
------------------------------------
Este script toma como insumo el output de 01_motor2_dataset.py
(motor2_pledge_month.parquet), que YA resolvió el anti-leakage (merge_asof con
allow_exact_matches=False) y ya trae la columna contact_id. No recalcula fechas
de cancelación desde cero: agrega lo que ya está validado.

Las reglas de agregación viven en scripts/motor2_donante_common.py, compartidas
con 04_score_motor2_donante.py para que el snapshot mensual se agregue
exactamente igual que el dataset de entrenamiento.

ENTRADA / SALIDA
----------------
Lee:    data/features/motor2_pledge_month.parquet   (con contact_id)
        data/staging/stg_payment_attempts.parquet   (solo fallback, si el
                                                     parquet no trae contact_id)
Genera: data/features/motor2_donante_month.parquet  (NUEVO — no pisa nada de
                                                     producción pledge-level)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # agrega scripts/
from motor2_donante_common import adjuntar_contact_id, agregar_a_donante

ROOT = Path(__file__).resolve().parents[2]
MOTOR2_PLEDGE = ROOT / "data/features/motor2_pledge_month.parquet"
STAGING = ROOT / "data/staging/stg_payment_attempts.parquet"
OUT = ROOT / "data/features/motor2_donante_month.parquet"


def main():
    if not MOTOR2_PLEDGE.exists():
        raise SystemExit(
            f"\n[ERROR] No existe {MOTOR2_PLEDGE}\n"
            f"        Genera primero el dataset pledge-level:\n"
            f"          python3 scripts/02_features_motor2/01_motor2_dataset.py\n")

    print("[1/3] Cargando dataset pledge-level (ya con anti-leakage resuelto)...")
    pledge_df = pd.read_parquet(MOTOR2_PLEDGE)
    print(f"  Filas pledge-level: {len(pledge_df):,}  |  "
          f"pledges: {pledge_df['pledge_id'].nunique():,}")

    print("[2/3] Resolviendo contact_id...")
    df = adjuntar_contact_id(pledge_df, STAGING)
    print(f"  Filas con contact_id: {len(df):,}  |  "
          f"donantes: {df['contact_id'].nunique():,}")

    print("[3/3] Agregando a nivel (contact_id, snapshot_ts)...")
    agregado = agregar_a_donante(df, con_label=True)

    print(f"  Filas donante-level: {len(agregado):,}  "
          f"(vs {len(df):,} pledge-level → reducción de "
          f"{100 * (1 - len(agregado) / len(df)):.1f}%)")
    print(f"  Pledges por donante-snapshot — media: "
          f"{agregado['n_pledges_activos'].mean():.3f}  "
          f"máx: {int(agregado['n_pledges_activos'].max())}")
    print(f"  Tasa churn 30d — pledge-level: {df['label_churn_30d'].mean():.4f}  "
          f"donante-level: {agregado['label_churn_30d'].mean():.4f}")
    print("  (se espera que la tasa donante-level sea MENOR o igual: cancelar TODOS "
          "los pledges es un evento más raro que cancelar UNO)")

    # Cuántos donantes-snapshot tienen más de un pledge: dimensiona cuánto
    # cambia realmente el problema. Si fuera ~0%, el modelo de donante sería
    # equivalente al de pledge y el cambio no valdría la pena.
    multi = agregado["n_pledges_activos"] > 1
    print(f"  Donante-snapshots con >1 pledge activo: {int(multi.sum()):,} "
          f"({100 * multi.mean():.2f}%)")
    if multi.any():
        print(f"  Tasa churn en ESE subconjunto (donde el cambio importa): "
              f"{agregado.loc[multi, 'label_churn_30d'].mean():.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    agregado.to_parquet(OUT, index=False)
    print(f"\n  Por año:")
    print(agregado.groupby(agregado["snapshot_ts"].dt.year).agg(
        n=("label_churn_30d", "size"), tasa=("label_churn_30d", "mean")))
    print(f"\n[OK] {OUT}")
    print(f"  Columnas ({len(agregado.columns)}): {agregado.columns.tolist()}")


if __name__ == "__main__":
    main()
