#!/usr/bin/env python3
"""
00_diagnostico_donante.py — Diagnóstico previo al cambio Motor 2 → nivel donante

CORRER ESTO PRIMERO, antes de tocar cualquier script de producción. Responde
tres preguntas que determinan todo el diseño de las features nuevas:

  1. ¿Cuántos pledges tiene un donante típico? (¿el problema es simple 1:1,
     o hay donantes con varios pledges activos a la vez?)
  2. ¿El mapeo pledge_id → contact_id es estable? (un pledge NUNCA debería
     cambiar de dueño a mitad de camino; si eso pasa, hay un problema de
     calidad de datos que hay que resolver antes de seguir)
  3. ¿Qué tan seguido un donante tiene pledges con estados mixtos (uno activo,
     otro ya cancelado) en el mismo snapshot? Esto dimensiona qué tan distinto
     va a ser el label a nivel donante vs. el label a nivel pledge actual.

NO escribe nada en production ni modifica ningún parquet. Solo imprime un
reporte. Ejecutar desde la raíz del entregable:

    python3 docs/00_diagnostico_donante.py

Lee:
    data/staging/stg_payment_attempts.parquet   (columnas: pledge_id, contact_id)
    data/features/motor2_pledge_month.parquet   (dataset pledge-level actual)
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "data/staging/stg_payment_attempts.parquet"
MOTOR2_PLEDGE = ROOT / "data/features/motor2_pledge_month.parquet"


def main():
    print("=" * 70)
    print("  DIAGNÓSTICO: cardinalidad pledge ↔ donante")
    print("=" * 70)

    # === 1. Mapeo pledge_id -> contact_id desde staging ===
    print("\n[1/4] Cargando staging (pledge_id, contact_id)...")
    stg = pd.read_parquet(STAGING, columns=["pledge_id", "contact_id"])
    print(f"  Filas (intentos) en staging: {len(stg):,}")

    # === 2. ¿Es estable el mapeo pledge_id -> contact_id? ===
    # Un pledge pertenece a UN donante. Si en el staging aparece con más de un
    # contact_id distinto entre sus intentos, algo está mal aguas arriba
    # (error de captura en el CRM, o un pledge reasignado de dueño).
    print("\n[2/4] Verificando estabilidad del mapeo pledge_id -> contact_id...")
    mapeo = stg.drop_duplicates(subset=["pledge_id", "contact_id"])
    contacts_por_pledge = mapeo.groupby("pledge_id")["contact_id"].nunique()
    pledges_inconsistentes = contacts_por_pledge[contacts_por_pledge > 1]
    n_incons = len(pledges_inconsistentes)
    print(f"  Pledges únicos: {contacts_por_pledge.shape[0]:,}")
    if n_incons == 0:
        print("  [OK] Todos los pledges tienen un único contact_id. Mapeo estable.")
    else:
        pct = 100 * n_incons / contacts_por_pledge.shape[0]
        print(f"  [AVISO] {n_incons:,} pledges ({pct:.2f}%) tienen MÁS DE UN contact_id "
              f"distinto en su historial de intentos.")
        print(f"  Ejemplos (primeros 10 pledge_id afectados):")
        print(f"  {pledges_inconsistentes.head(10).index.tolist()}")
        print(f"  → Hay que decidir una regla de resolución (ej: usar el contact_id "
              f"más reciente, o el más frecuente) ANTES de construir el label de donante.")

    # Mapeo final resuelto: un contact_id por pledge (el más frecuente en su historial,
    # para blindar contra el caso inconsistente de arriba).
    pledge_to_contact = (
        stg.groupby(["pledge_id", "contact_id"]).size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates(subset=["pledge_id"], keep="first")
        [["pledge_id", "contact_id"]]
    )

    # === 3. Cardinalidad: pledges por donante ===
    print("\n[3/4] Cardinalidad pledges por donante...")
    pledges_por_donante = pledge_to_contact.groupby("contact_id")["pledge_id"].nunique()
    print(f"  Donantes únicos: {pledges_por_donante.shape[0]:,}")
    print(f"  Distribución de #pledges por donante:")
    print(pledges_por_donante.value_counts().sort_index().head(15).to_string())
    multi = (pledges_por_donante > 1).sum()
    pct_multi = 100 * multi / pledges_por_donante.shape[0]
    print(f"\n  Donantes con MÁS DE 1 pledge (histórico, no necesariamente simultáneo): "
          f"{multi:,} ({pct_multi:.2f}%)")
    if pct_multi < 5:
        print("  → Caso simple: la gran mayoría es 1:1. El cambio a nivel donante va a "
              "mover poco el dataset, pero sigue siendo la forma correcta de hacerlo.")
    else:
        print("  → Caso no trivial: una fracción relevante de donantes tiene varios "
              "pledges. El label a nivel donante SÍ va a diferir material del label "
              "a nivel pledge — esto es justo lo que estamos corrigiendo.")

    # === 4. Pledges simultáneamente activos por donante (si el dataset ya existe) ===
    if MOTOR2_PLEDGE.exists():
        print("\n[4/4] Pledges ACTIVOS simultáneamente por donante y snapshot "
              "(usando motor2_pledge_month.parquet)...")
        m2 = pd.read_parquet(MOTOR2_PLEDGE, columns=["pledge_id", "snapshot_ts", "label_churn_30d"])
        m2 = m2.merge(pledge_to_contact, on="pledge_id", how="left")
        sin_contacto = m2["contact_id"].isna().sum()
        if sin_contacto:
            print(f"  [AVISO] {sin_contacto:,} filas de motor2_pledge_month.parquet no "
                  f"encontraron contact_id en el staging (pledge sin intentos, o "
                  f"pledge_id desalineado entre fuentes). Revisar antes de continuar.")
        activos_por_snap = m2.dropna(subset=["contact_id"]).groupby(
            ["contact_id", "snapshot_ts"])["pledge_id"].nunique()
        print(f"  Distribución de #pledges activos simultáneos (donante × snapshot):")
        print(activos_por_snap.value_counts().sort_index().head(10).to_string())

        # Donde el label a nivel donante (min por grupo) difiere del label pledge-level
        # promedio del grupo — para dimensionar cuánto cambia el problema.
        m2v = m2.dropna(subset=["contact_id"]).copy()
        grp = m2v.groupby(["contact_id", "snapshot_ts"])["label_churn_30d"]
        label_donante = grp.transform("min")
        multi_pledge_mask = m2v.groupby(["contact_id", "snapshot_ts"])["pledge_id"].transform("nunique") > 1
        diff = (m2v["label_churn_30d"] != label_donante) & multi_pledge_mask
        print(f"\n  Filas (pledge, snapshot) con >1 pledge activo donde el label "
              f"individual difiere del label de donante: {int(diff.sum()):,} "
              f"de {int(multi_pledge_mask.sum()):,} filas multi-pledge "
              f"({100*diff.sum()/max(multi_pledge_mask.sum(),1):.2f}%).")
        print("  → Esas son exactamente las filas donde el modelo actual (pledge-level) "
              "está prediciendo mal el evento que de verdad le importa al negocio "
              "(el donante, no el pledge individual).")
    else:
        print(f"\n[4/4] {MOTOR2_PLEDGE} no existe todavía — omito el cruce con el "
              f"dataset actual. Corre 01_motor2_dataset.py primero si quieres esta parte.")

    print("\n" + "=" * 70)
    print("  FIN DEL DIAGNÓSTICO")
    print("=" * 70)


if __name__ == "__main__":
    main()
