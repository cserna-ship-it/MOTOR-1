#!/usr/bin/env python3
"""
01_clean_bank_normalization.py — Features Motor 1 (paso 1 de 3)

PROPÓSITO
---------
Limpiar la columna `bank_name` del staging para convertirla en una
categórica usable. Es necesaria porque los datos crudos tienen ~571
variantes del nombre del banco — pero solo ~40 bancos reales en Colombia.
La discrepancia viene de tres fuentes:

1. **Mojibake (encoding roto)**: el export pasó por una conversión
   latin-1 ↔ utf-8 que rompió los caracteres con tilde.
   Ejemplos reales encontrados:
       "BANCO DE BOGOTÁ"       (original, correcto)
       "BANCO DE BOGOTÃ�"      (1 conversión rota)
       "BANCO DE BOGOTÃƒï¿½"   (2 conversiones rotas anidadas)

2. **Variantes ortográficas**: con/sin "SA", con/sin "S.A.", con/sin
   paréntesis, etc.
       "BANCOLOMBIA" / "BANCOLOMBIA SA" / "BANCOLOMBIA S.A." / "BANCOLOMBIA S.A"
       "ITAÚ" / "ITAÚ (HELM BANK)" / "ITAU HELM BANK"

3. **Fusiones bancarias históricas**: Helm Bank fue absorbido por Itaú
   y Colpatria por Scotiabank — pero registros viejos siguen con el
   nombre anterior.

ESTRATEGIA EN 4 PASOS
---------------------
1. Decodificar mojibake aplicando hasta 4 iteraciones de latin-1 ↔ utf-8.
   (Algunas variantes están "doble-mojibakeadas" y requieren múltiples passes).
2. Normalizar Unicode NFKD para descomponer caracteres y eliminar
   diacríticos (Á → A, Ñ → N, etc.).
3. Limpiar puntuación y consolidar abreviaciones "S.A." / "S.A" / "SA".
4. Aplicar `CANONICAL_MAP` (mapeo manual) para los casos que las reglas
   automáticas no consolidan.

ENTRADA / SALIDA
----------------
Lee:    data/staging/stg_payment_attempts.parquet (solo columna bank_name)
Genera:
    - data/features/bank_canonical.parquet   → 1 col, paralela al staging
    - data/features/bank_canonical_mapping.json → mapping completo para auditoría

USO EN EL MODELO
----------------
`bank_canonical` se usa como feature categórica en Motor 1. La diferencia
entre bancos en tasa de éxito es significativa (rango observado 70%-96%),
por lo que esta normalización aporta lift real al modelo.

REPLICABILIDAD
--------------
Si el cliente reporta un banco no consolidado correctamente, agregarlo
manualmente a `CANONICAL_MAP` y re-correr. El JSON de auditoría permite
revisar todas las normalizaciones aplicadas.
"""
from pathlib import Path
import re
import json
import unicodedata
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STAGING = ROOT / "data/staging/stg_payment_attempts.parquet"
OUT_DIR = ROOT / "data/features"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bank(name):
    """
    Aplica la canonicalización completa a un string de nombre de banco.

    Devuelve "UNKNOWN" para nulos, strings vacíos, o casos donde la
    limpieza deja un string vacío.

    Las 3 fases (numeradas en el código):
    1. Fix de mojibake hasta 4 iteraciones — algunos strings están
       "doble-mojibakeados" (la conversión rota ocurrió 2 veces) y
       requieren múltiples pases para estabilizarse. La detección se
       basa en marcadores típicos del mojibake latin-1↔utf-8: "Ã", "ï¿½",
       "Â", "Å".
    2. NFKD + strip de diacríticos: descompone "Á" → "A" + acento agudo
       (combining char), luego elimina el combining char.
    3. Limpieza de puntuación: uppercases, reemplaza comas, consolida
       las variantes "S.A.", "S.A", "S A", "SA" → "SA"; análogo "SAS".
       Finalmente quita "SA" al final porque no aporta información
       discriminatoria entre bancos.
    """
    if not isinstance(name, str) or len(name) == 0:
        return "UNKNOWN"
    s = name

    # === 1. Fix iterativo de mojibake ===
    # El loop está acotado a 4 iteraciones para evitar bucles infinitos
    # en casos patológicos. En la práctica con 2-3 pasadas suele bastar.
    for _ in range(4):
        if "Ã" in s or "ï¿½" in s or "Â" in s or "Å" in s:
            try:
                # Conversión inversa: tomar el string como si fuera latin-1
                # encoded y reinterpretarlo como utf-8. Esto corrige el
                # mojibake clásico de "doble-encoding".
                fixed = s.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
                if fixed and fixed != s:
                    s = fixed
                else:
                    # No cambió → ya estabilizó, salimos.
                    break
            except Exception:
                break
        else:
            # No quedan marcadores de mojibake → string ya está limpio.
            break

    # === 2. Normalización Unicode + remoción de diacríticos ===
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", errors="ignore").decode("ascii")

    # === 3. Limpieza de puntuación y abreviaciones de sociedad ===
    s = s.upper().strip()
    s = re.sub(r",", " ", s)
    # Unificar variantes de "S.A." (sociedad anónima) — todas pasan a "SA"
    s = re.sub(r"\s+S\.?\s*A\.?(\s|$)", " SA ", s)
    # Análogo para "S.A.S." (sociedad anónima simplificada)
    s = re.sub(r"\s+S\.?\s*A\.?\s*S\.?(\s|$)", " SAS ", s)
    # Eliminar todo carácter que no sea alfanumérico, espacio, paréntesis o guión
    s = re.sub(r"[^A-Z0-9\s\(\)\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Quitar "SA" final: no discrimina entre bancos. "BANCOLOMBIA" y
    # "BANCOLOMBIA SA" son el mismo banco.
    s = re.sub(r"\bSA\b\s*$", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()

    return s if s else "UNKNOWN"


# Mapeo manual para casos que la limpieza algorítmica NO logra consolidar.
# Se aplica DESPUÉS de normalize_bank() — es decir, sobre strings ya
# limpios de mojibake y diacríticos. Las claves son strings ya normalizados.
#
# Casos cubiertos:
#   - Truncamientos del nombre original (ej: "BANCO BOGOT" ← "BANCO BOGOTÁ"
#     que perdió la última letra después del mojibake fix).
#   - Fusiones bancarias: HELM BANK → ITAÚ, COLPATRIA → SCOTIABANK.
#   - Variantes específicas de Bilbao Vizcaya (largo → BBVA).
#
# Cuando aparezca un banco nuevo no consolidado, agregar la entrada aquí
# y re-correr el script. El JSON de auditoría
# (bank_canonical_mapping.json) facilita identificar variantes pendientes.
CANONICAL_MAP = {
    "BANCO BOGOTA": "BANCO DE BOGOTA",
    "BANCO BOGOT": "BANCO DE BOGOTA",
    "BANCO DE BOGOT": "BANCO DE BOGOTA",
    "BANCO DE BOGOTAI12": "BANCO DE BOGOTA",
    "BANCO BOGOTA HELM BANK": "BANCO DE BOGOTA",
    "ITAU": "ITAU",
    "ITAAS": "ITAU",
    "ITAAA": "ITAU",
    "ITA": "ITAU",
    "ITAS": "ITAU",
    "ITAU HELM BANK": "ITAU",
    "ITAU (HELM BANK)": "ITAU",
    "ITAAS (HELM BANK)": "ITAU",
    "ITAAA (HELM BANK)": "ITAU",
    "ITA (HELM BANK)": "ITAU",
    "ITAS (HELM BANK)": "ITAU",
    "BANCOLOMBIA SA": "BANCOLOMBIA",
    "BANCO DAVIVIENDA": "BANCO DAVIVIENDA",
    "BANCO BILBAO VIZCAYA ARGENTARIA COLOMBIA BBVA COLOMBIA": "BBVA COLOMBIA",
    "BANCO BILBAO VIZCAYA ARGENTARIA COLOMBIA": "BBVA COLOMBIA",
    "BANCO BILBAO VIZCAYA ARGENTARIA (BBVA)": "BBVA COLOMBIA",
    "BANCO CAJA SOCIAL BCSC": "BANCO CAJA SOCIAL",
    "SCOTIABANK COLPATRIA": "SCOTIABANK COLPATRIA",
    "BANCO COLPATRIA": "SCOTIABANK COLPATRIA",
    "DAVIPLATA": "DAVIPLATA",
}


def main():
    print("[1/3] Cargando staging...")
    df = pd.read_parquet(STAGING, columns=["bank_name"])
    print(f"  filas: {len(df):,}")
    print(f"  variantes originales: {df['bank_name'].nunique():,}")

    print("[2/3] Aplicando normalización...")
    unique_names = df["bank_name"].dropna().unique()
    mapping = {name: normalize_bank(name) for name in unique_names}
    # Aplicar canonical map
    for k, v in mapping.items():
        if v in CANONICAL_MAP:
            mapping[k] = CANONICAL_MAP[v]

    df["bank_canonical"] = df["bank_name"].map(mapping).fillna("UNKNOWN").astype("category")
    print(f"  variantes después: {df['bank_canonical'].nunique():,}")

    # Top 20 antes/después
    print("\n  Top 15 después de normalización:")
    print(df["bank_canonical"].value_counts().head(15))

    # Guardar mapping para auditoría
    with open(OUT_DIR / "bank_canonical_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    # Verificar fusión "BANCO DE BOGOTA"
    print("\n  Verificación BANCO DE BOGOTA:")
    test_names = ["BANCO DE BOGOTÁ", "BANCO DE BOGOTÃ�", "BANCO DE BOGOTÃƒï¿½",
                  "BANCO DE BOGOTÃƒ", "BANCO DE BOGOTÃ", "BANCO BOGOTÁ"]
    for n in test_names:
        if n in mapping:
            print(f"    '{n}' → '{mapping[n]}'")

    # También verificar Itaú
    print("\n  Verificación ITAU:")
    for n in ["ITAÚ", "ITAÃš", "ITAÃƒÅ¡", "ITAÚ (HELM BANK)", "ITAÃš (HELM BANK)", "ITAÚ* (HELM BANK)"]:
        if n in mapping:
            print(f"    '{n}' → '{mapping[n]}'")

    print("\n[3/3] Guardando staging enriquecido...")
    out = df[["bank_canonical"]].copy()
    out.index.name = "row_idx"
    out.to_parquet(OUT_DIR / "bank_canonical.parquet")
    print(f"[OK] {OUT_DIR / 'bank_canonical.parquet'}")


if __name__ == "__main__":
    main()
