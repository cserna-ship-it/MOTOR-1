#!/usr/bin/env python3
"""
00_unify_raw_data.py — ETAPA 0 del pipeline (preprocesamiento)

PROPÓSITO
---------
Convertir los CSV crudos exportados de Salesforce en parquets unificados por
entidad de negocio. Es la única etapa que toca los archivos del cliente; todo
el resto del pipeline trabaja sobre los parquets producidos aquí.

QUÉ HACE
--------
1. Lee los 9 CSV de `raw/` (4.2 GB en total) con manejo robusto de encodings
   (utf-8 / utf-8-sig / latin1 / cp1252 — Salesforce a veces exporta en cp1252).
2. Para cada entidad, normaliza nombres de columna a snake_case (ver función
   `snake_case` abajo) — paso CRÍTICO porque los CSV mezclan estilos:
       · payments.csv usa Title Case con espacios ("Payment ID", "Bank Name")
       · contact/opportunity/pledge/allocation usan estilo Salesforce
         con sufijos __c ("npsp__Status__c", "Gender__c")
   Después de snake_case, todos los nombres quedan en formato uniforme.
3. Une múltiples particiones (ej: `Opportunity_1_2024.csv` +
   `Opportunity_2_2024.csv` + ... 2025) en un solo parquet por entidad.
4. Agrega 2 columnas de trazabilidad a cada fila:
       · source_file: nombre del CSV original
       · source_year: año extraído del nombre del archivo
5. Guarda parquet + CSV (para inspección manual) y emite un reporte JSON con
   métricas básicas de calidad (filas, columnas, % nulos top 20,
   unicidad de IDs, valores de columnas de status).

REGLAS DE NEGOCIO IMPLÍCITAS
----------------------------
- NO se eliminan filas en esta etapa. Todo lo que viene del CRM se mantiene
  (filtros y limpieza ocurren en `01_create_staging.py` y posteriores).
- Si una columna existe en una partición pero no en otra (drift de esquema
  entre exports), se rellena con NA en las particiones donde falta — esto
  evita perder filas por joins desalineados.
- Valores `"null"`, `"NA"`, `"#N/A"`, etc. se convierten a `pd.NA` para
  unificar la representación de nulos (Salesforce los exporta de varias
  formas según el tipo de campo).

ENTRADA
-------
- Directorio `raw/` con los 9 CSVs originales del cliente.

SALIDA
------
- `data/unified/<entidad>/<entidad>_unified.parquet` (5 parquets)
- `data/unified/<entidad>/<entidad>_unified.csv` (para inspección)
- `data/unified/<entidad>/<entidad>_unification_report.json` (reporte por entidad)
- `data/unified/master_unification_report.json` (reporte maestro)

Uso:
    python 00_unify_raw_data.py --raw-dir ./raw --out-dir ./data/unified

Estructura esperada:
    raw/
      Contact-3_16_2026.csv
      Opportunity_{1,2}_{2024,2025}.csv
      npe03__Recurring_Donation__Pledge.csv
      payments 2024.csv
      payments_2025.csv
      sescore__Recurring_Donation_Allocation__RDA.csv

REPLICABILIDAD
--------------
Si llegan datos nuevos del cliente (ej: payments_2026.csv), basta con:
1. Agregarlo a la lista correspondiente en FILE_GROUPS abajo, o
2. Si conserva el patrón de nombre (payments_<año>.csv), agregarlo al
   directorio raw/ y actualizar FILE_GROUPS["payments"].
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


FILE_GROUPS = {
    "contact": [
        "raw/Contact-3_16_2026.csv",
    ],
    "opportunity": [
        "raw/Opportunity_1_2024.csv",
        "raw/Opportunity_2_2024.csv",
        "raw/Opportunity_1_2025.csv",
        "raw/Opportunity_2_2025.csv",
    ],
    "pledge": [
        "raw/npe03__Recurring_Donation__Pledge.csv",
    ],
    "payments": [
        "raw/payments 2024.csv",
        "raw/payments_2025.csv",
    ],
    "allocation": [
        "raw/sescore__Recurring_Donation_Allocation__RDA.csv",
    ],
}

DEFAULT_ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]


def snake_case(name: str) -> str:
    """
    Normaliza un nombre de columna a snake_case.

    Reglas aplicadas en orden:
    1. Quita espacios al inicio/fin.
    2. Reemplaza cualquier secuencia de caracteres no-alfanuméricos por un solo '_'.
       Esto unifica espacios, puntos, paréntesis, guiones, etc.
    3. Colapsa secuencias de múltiples '_' a un solo '_'. Crítico para columnas
       de Salesforce con doble guión bajo ("sescore__Type__c" → "sescore_type_c").
    4. Inserta '_' entre transiciones minúscula→Mayúscula para partir camelCase
       ("PaymentMethod" → "Payment_Method").
    5. Strip de '_' en bordes + lowercase final.

    Ejemplos:
        "Payment ID"                      → "payment_id"
        "CRM Donation ID"                 → "crm_donation_id"
        "npe03__Recurring_Donation__c"    → "npe03_recurring_donation_c"
        "sescore__Processed_Date_Time__c" → "sescore_processed_date_time_c"
        "Gender__c"                       → "gender_c"
        "MailingState"                    → "mailing_state"

    IMPORTANTE: esta función es la fuente de verdad para los nombres de columna
    que usan los scripts aguas abajo. Cambiar la lógica acá rompe TODO el resto
    del pipeline.
    """
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"__+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.strip("_").lower()
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unificar CSVs crudos por entidad.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Directorio donde están los CSV originales.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directorio de salida para archivos unificados.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_robust(path: Path) -> pd.DataFrame:
    """
    Lee un CSV intentando varios encodings hasta encontrar uno que funcione.

    Por qué hace falta esto: los exports de Salesforce de UNICEF Colombia
    contienen mojibake conocido (especialmente en `Bank Name` con caracteres
    como "BANCO DE BOGOTÁ" → "BANCO DE BOGOTÃ�"). Probamos en orden:
        utf-8 → utf-8-sig (BOM) → latin1 → cp1252 (Windows-1252).

    Nota: dtype=str fuerza a leer todas las columnas como string. Esto es
    deliberado — la normalización de tipos ocurre aguas abajo, donde sabemos
    qué columnas son numéricas, fechas, etc. Leer todo como string evita
    que pandas haga inferencia incorrecta (ej: leer "00012" como int).
    """
    last_error = None
    for enc in DEFAULT_ENCODINGS:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                encoding=enc,
                low_memory=False,
            )
        except Exception as e:
            last_error = e
    raise RuntimeError(f"No se pudo leer {path.name}. Último error: {last_error}")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las dos normalizaciones globales a un DataFrame recién leído:
    1. Renombra columnas a snake_case (con deduplicación si hay colisiones).
    2. Convierte representaciones varias de "nulo" (string "null", "NA",
       "#N/A", etc.) a pd.NA real para que aguas abajo `.isna()` funcione.

    REGLA DE DEDUPLICACIÓN (columnas):
    Si dos columnas originales colapsan al mismo snake_case (ej: "Payment_id"
    y "PAYMENT ID" → ambos "payment_id"), se mantiene la primera y a las
    siguientes se les añade sufijo _1, _2, ... Esto NO debería ocurrir en
    los CSVs actuales pero protege contra exports con duplicados.

    REGLA DE NULOS:
    Se trata como nulo TODO valor cuya representación textual sea (case-
    insensitive, ya en lowercase tras snake_case): vacío, "null", "none",
    "nan", "na", "n/a", "#n/a". También se atrapan los artefactos
    "<NA>", "None", "NaT" que pandas puede dejar al leer.
    """
    df = df.copy()

    # ---- 1. Renombre a snake_case ----
    new_cols = [snake_case(c) for c in df.columns]

    # Si una columna no tiene nombre (ej: la primera columna "_" de los
    # exports SF) queda string vacío tras snake_case. Le damos un nombre
    # sintético para evitar colisiones.
    new_cols = [c if c else f"unnamed_{i}" for i, c in enumerate(new_cols)]

    # Deduplicación: si dos columnas distintas colapsan al mismo nombre
    # (caso defensivo), agrego sufijo _1, _2, ...
    seen = {}
    final_cols = []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            final_cols.append(c)
        else:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")

    df.columns = final_cols

    # ---- 2. Unificación de representaciones de NULL ----
    # Salesforce exporta nulos como "", "null", "NA", "#N/A" según el tipo
    # de campo. Los traemos todos al mismo pd.NA para que el resto del
    # pipeline pueda usar .isna() de forma confiable.
    null_like = {"", "null", "none", "nan", "na", "n/a", "#n/a"}
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(list(null_like), pd.NA)
            df.loc[df[col].isin(["<NA>", "None", "NaT"]), col] = pd.NA
    return df


def infer_source_year(filename: str) -> str | None:
    match = re.search(r"(20\d{2})", filename)
    return match.group(1) if match else None


def add_file_metadata(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.copy()
    df["source_file"] = filename
    df["source_year"] = infer_source_year(filename)
    return df


def align_columns(dfs: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], List[str]]:
    all_cols = sorted(set().union(*[set(df.columns) for df in dfs]))
    aligned = []
    for df in dfs:
        missing = [c for c in all_cols if c not in df.columns]
        for c in missing:
            df[c] = pd.NA
        aligned.append(df[all_cols].copy())
    return aligned, all_cols


def basic_profile(df: pd.DataFrame) -> Dict:
    """
    Genera un perfil rápido de calidad para una entidad unificada.

    Métricas incluidas en el JSON de salida:
        - rows / columns: tamaño total
        - column_names: lista completa (sirve como catálogo)
        - null_pct_top20: top 20 columnas con MÁS nulos (% sobre filas)
        - exact_duplicate_rows: número de filas 100% idénticas
        - id_uniqueness: para cada columna ID o *_id (hasta 15):
              non_null, distinct, duplicated_non_null
          → permite detectar PKs corruptas en el export
        - status_candidates: para cada columna que contenga "status" o "state"
          (hasta 10): top 20 valores con sus conteos
          → permite detectar valores nuevos del CRM que romperían el target
            del modelo (ej: si aparece "Completed" en vez de "Paid")

    Este reporte es la primera línea de defensa contra cambios silenciosos en
    los datos cuando se re-corra el pipeline con datos nuevos en el futuro.
    """
    profile = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "null_pct_top20": {},
        "exact_duplicate_rows": int(df.duplicated().sum()),
    }

    null_pct = (df.isna().mean() * 100).sort_values(ascending=False).head(20)
    profile["null_pct_top20"] = {k: round(float(v), 2) for k, v in null_pct.items()}

    id_candidates = [c for c in df.columns if c == "id" or c.endswith("_id")]
    id_uniqueness = {}
    for c in id_candidates[:15]:
        non_null = df[c].dropna()
        if len(non_null) > 0:
            id_uniqueness[c] = {
                "non_null": int(non_null.shape[0]),
                "distinct": int(non_null.nunique(dropna=True)),
                "duplicated_non_null": int(non_null.duplicated().sum()),
            }
    profile["id_uniqueness"] = id_uniqueness

    status_candidates = [c for c in df.columns if "status" in c or "state" in c]
    status_values = {}
    for c in status_candidates[:10]:
        vals = (
            df[c]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts(dropna=False)
            .head(20)
            .to_dict()
        )
        status_values[c] = {str(k): int(v) for k, v in vals.items()}
    profile["status_candidates"] = status_values

    return profile


def save_outputs(df: pd.DataFrame, entity_name: str, out_dir: Path) -> Dict[str, str]:
    entity_dir = out_dir / entity_name
    ensure_dir(entity_dir)

    parquet_path = entity_dir / f"{entity_name}_unified.parquet"
    csv_path = entity_dir / f"{entity_name}_unified.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    return {
        "parquet": str(parquet_path),
        "csv": str(csv_path),
    }


def unify_entity(entity_name: str, filenames: List[str], raw_dir: Path, out_dir: Path) -> Dict:
    """
    Pipeline completo para UNA entidad (payments, opportunity, pledge, etc.):
        1. Verifica que todos los archivos esperados existan.
        2. Lee cada partición, normaliza columnas/nulos, agrega metadata.
        3. Alinea las columnas entre particiones (las que faltan en alguna
           se rellenan con NA — esto preserva todas las filas y evita
           sorpresas si Salesforce cambia el esquema entre exports).
        4. Concatena, guarda en parquet+csv, perfila y emite reporte JSON.

    El reporte JSON por entidad es el contrato verificable con el cliente:
    si los conteos cambian inesperadamente entre corridas, hay drift de
    datos que requiere atención antes de re-entrenar.
    """
    # Failure-fast si falta alguno de los CSVs esperados — preferible a
    # generar un parquet incompleto silenciosamente.
    missing_files = [f for f in filenames if not (raw_dir / f).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Faltan archivos para {entity_name}: {', '.join(missing_files)}"
        )

    original_dfs = []
    file_summaries = []

    # Cada partición se procesa de forma independiente y aislada antes
    # de concatenar. Esto permite que una partición con esquema distinto
    # no contamine al resto.
    for filename in filenames:
        path = raw_dir / filename
        df = read_csv_robust(path)
        raw_shape = df.shape

        df = normalize_df(df)
        df = add_file_metadata(df, filename)  # agrega source_file + source_year

        original_dfs.append(df)
        file_summaries.append(
            {
                "file": filename,
                "rows": int(raw_shape[0]),
                "columns_after_normalization": int(df.shape[1]),
                "source_year": infer_source_year(filename),
            }
        )

    # Alineación de columnas entre particiones: union, no intersección.
    # Si la partición 2024 tiene una columna que la 2025 no tiene (o vice-
    # versa), la columna queda en el parquet final con NA donde falta.
    # Esto preserva el máximo de información sin perder filas.
    aligned_dfs, all_cols = align_columns(original_dfs)
    unified = pd.concat(aligned_dfs, ignore_index=True)

    outputs = save_outputs(unified, entity_name, out_dir)
    profile = basic_profile(unified)

    entity_report = {
        "entity": entity_name,
        "input_files": file_summaries,
        "final_rows": int(unified.shape[0]),
        "final_columns": int(unified.shape[1]),
        "all_columns_count": len(all_cols),
        "outputs": outputs,
        "profile": profile,
    }

    report_path = out_dir / entity_name / f"{entity_name}_unification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(entity_report, f, ensure_ascii=False, indent=2)

    return entity_report


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    ensure_dir(out_dir)

    master_report = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "entities": {},
    }

    for entity_name, filenames in FILE_GROUPS.items():
        print(f"\n[INFO] Unificando entidad: {entity_name}")
        report = unify_entity(
            entity_name=entity_name,
            filenames=filenames,
            raw_dir=raw_dir,
            out_dir=out_dir,
        )
        master_report["entities"][entity_name] = report
        print(
            f"[OK] {entity_name}: {report['final_rows']:,} filas, "
            f"{report['final_columns']} columnas"
        )

    master_report_path = out_dir / "master_unification_report.json"
    with open(master_report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, ensure_ascii=False, indent=2)

    print("\n[FIN] Unificación completada.")
    print(f"Reporte maestro: {master_report_path}")


if __name__ == "__main__":
    main()
