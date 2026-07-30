"""
quality.py
-----------
Módulo de Auditoría de Calidad de Datos - Fase 1
Proyecto: TechLogistics S.A.S. - Challenge 02 (EAFIT)

Funciones puras y reutilizables para:
  - Calcular nulidad por columna
  - Detectar duplicados (por fila completa y por llave primaria)
  - Detectar outliers numéricos vía rango intercuartílico (IQR)
  - Generar un "Health Score" por dataset

Este módulo será importado por la app de Streamlit (Fase 2+) sin
mezclar lógica de UI con lógica de negocio, siguiendo el requisito
de código modular del challenge.
"""

import pandas as pd
import numpy as np


class QualityReportError(Exception):
    """Error de negocio calculando una métrica de calidad de datos."""


# ---------------------------------------------------------------------------
# 1. NULIDAD
# ---------------------------------------------------------------------------
def null_report(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve % y conteo de nulos por columna, ordenado descendente."""
    if df.empty:
        return pd.DataFrame(columns=["columna", "nulos", "pct_nulidad"])
    nulls = df.isna().sum()
    pct = (nulls / len(df) * 100).round(2)
    out = pd.DataFrame({
        "columna": df.columns,
        "nulos": nulls.values,
        "pct_nulidad": pct.values
    }).sort_values("pct_nulidad", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 2. DUPLICADOS
# ---------------------------------------------------------------------------
def duplicate_report(df: pd.DataFrame, id_col: str | None = None) -> dict:
    """
    Reporta duplicados a dos niveles:
      - full_row: filas 100% idénticas (copy-paste)
      - id_collision: mismo valor de llave primaria pero fila distinta
        (falla de generación de ID, NO necesariamente el mismo evento)
    """
    full_row_dupes = int(df.duplicated().sum())
    result = {
        "full_row_duplicates": full_row_dupes,
    }
    if id_col is not None and id_col in df.columns:
        id_dupe_mask = df[id_col].duplicated(keep=False)
        result["id_collisions_rows"] = int(id_dupe_mask.sum())
        result["id_collisions_unique_ids"] = int(df.loc[id_dupe_mask, id_col].nunique())
    return result


# ---------------------------------------------------------------------------
# 3. OUTLIERS (IQR)
# ---------------------------------------------------------------------------
def iqr_outliers(series: pd.Series) -> dict:
    """Calcula límites y conteo de outliers por método IQR (1.5x)."""
    s = series.dropna()
    if len(s) == 0:
        return {"q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "lower": np.nan, "upper": np.nan, "n_outliers": 0, "pct_outliers": 0.0}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return {
        "q1": round(q1, 2), "q3": round(q3, 2), "iqr": round(iqr, 2),
        "lower": round(lower, 2), "upper": round(upper, 2),
        "n_outliers": int(mask.sum()),
        "pct_outliers": round(mask.sum() / len(s) * 100, 2)
    }


def outlier_report(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Aplica iqr_outliers a cada columna numérica solicitada."""
    rows = []
    for col in numeric_cols:
        if col not in df.columns:
            raise QualityReportError(
                f"outlier_report: la columna '{col}' no existe en el dataframe."
            )
        stats = iqr_outliers(df[col])
        stats["columna"] = col
        rows.append(stats)
    if not rows:
        return pd.DataFrame(
            columns=["columna", "q1", "q3", "iqr", "lower", "upper", "n_outliers", "pct_outliers"]
        )
    return pd.DataFrame(rows)[
        ["columna", "q1", "q3", "iqr", "lower", "upper", "n_outliers", "pct_outliers"]
    ]


def outlier_report_by_group(df: pd.DataFrame, col: str, group_col: str) -> pd.DataFrame:
    """
    Igual que iqr_outliers, pero calculando el IQR de forma independiente
    dentro de cada grupo (p. ej. cada Categoria) en vez de sobre la columna
    completa. Es la versión metodológicamente más rigurosa: un valor puede
    ser normal para una categoría y anómalo para otra, y el IQR global no
    distingue eso -- puede diluir anomalías reales o marcar como "outlier"
    algo que es perfectamente normal dentro de su propio grupo.
    """
    if col not in df.columns:
        raise QualityReportError(f"outlier_report_by_group: la columna '{col}' no existe.")
    if group_col not in df.columns:
        raise QualityReportError(f"outlier_report_by_group: la columna '{group_col}' no existe.")

    rows = []
    for grupo, sub in df.groupby(group_col):
        stats = iqr_outliers(sub[col])
        stats["grupo"] = grupo
        stats["n"] = len(sub)
        rows.append(stats)
    if not rows:
        return pd.DataFrame(
            columns=["grupo", "n", "q1", "q3", "iqr", "lower", "upper",
                     "n_outliers", "pct_outliers"]
        )
    return pd.DataFrame(rows)[
        ["grupo", "n", "q1", "q3", "iqr", "lower", "upper", "n_outliers", "pct_outliers"]
    ].sort_values("pct_outliers", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. HEALTH SCORE
# ---------------------------------------------------------------------------
def health_score(df: pd.DataFrame, numeric_cols: list, id_col: str | None = None) -> dict:
    """
    Health Score simple (0-100) penalizado por:
      - nulidad promedio
      - % de duplicados (fila completa)
      - % promedio de outliers en columnas numéricas clave
    Este score es una heurística de negocio, no una métrica estadística formal;
    su propósito es comunicar salud relativa del dataset a la junta directiva.
    """
    if df.empty:
        return {"health_score": 0.0, "avg_null_pct": 0.0, "full_row_duplicate_pct": 0.0,
                "avg_outlier_pct": 0.0}

    null_rep = null_report(df)
    avg_null_pct = null_rep["pct_nulidad"].mean() if not null_rep.empty else 0.0
    dupes = duplicate_report(df, id_col)
    dupe_pct = dupes["full_row_duplicates"] / len(df) * 100

    outliers = outlier_report(df, numeric_cols)
    avg_outlier_pct = outliers["pct_outliers"].mean() if not outliers.empty else 0.0

    score = 100 - (avg_null_pct * 0.4) - (dupe_pct * 0.3) - (avg_outlier_pct * 0.3)
    score = max(0, min(100, round(score, 1)))

    return {
        "health_score": score,
        "avg_null_pct": round(avg_null_pct, 2),
        "full_row_duplicate_pct": round(dupe_pct, 2),
        "avg_outlier_pct": round(avg_outlier_pct, 2),
    }


# ---------------------------------------------------------------------------
# 5. TRAZABILIDAD DE INGRESOS (crudo -> limpio)
# ---------------------------------------------------------------------------
def revenue_reconciliation(tr_raw: pd.DataFrame, tr_clean: pd.DataFrame) -> dict:
    """
    Reconcilia el ingreso bruto del archivo crudo de transacciones contra el
    ingreso ya calculado sobre las transacciones limpias (Fase 1), para que la
    cifra final sea trazable hasta el archivo original (requisito de la Guía
    de Validación: "Integridad de Identidad / Merging").

    La diferencia entre ambos NO debería ser cero: se explica en su totalidad
    por la corrección documentada del centinela Cantidad_Vendida = -5
    (Fase 1), que en el archivo crudo resta ingreso de forma artificial en
    100 filas. Este cálculo usa exactamente la misma columna Cantidad_Vendida
    limpia que alimenta la Sola Fuente de Verdad, así que la reconciliación
    es exacta, no una aproximación.
    """
    ingreso_bruto_crudo = float((tr_raw["Precio_Venta_Final"] * tr_raw["Cantidad_Vendida"]).sum())
    ingreso_post_limpieza = float(
        (tr_clean["Precio_Venta_Final"] * tr_clean["Cantidad_Vendida"]).sum()
    )

    mask_centinela = tr_raw["Cantidad_Vendida"] == -5
    n_centinela = int(mask_centinela.sum())
    precio_filas_centinela = tr_raw.loc[mask_centinela, "Precio_Venta_Final"]
    cantidad_imputada = tr_clean.loc[mask_centinela, "Cantidad_Vendida"]
    efecto_correccion = float(
        (precio_filas_centinela * cantidad_imputada).sum()
        - (precio_filas_centinela * tr_raw.loc[mask_centinela, "Cantidad_Vendida"]).sum()
    )

    return {
        "ingreso_bruto_crudo_usd": round(ingreso_bruto_crudo, 2),
        "ingreso_post_limpieza_usd": round(ingreso_post_limpieza, 2),
        "diferencia_usd": round(ingreso_post_limpieza - ingreso_bruto_crudo, 2),
        "n_filas_centinela_cantidad": n_centinela,
        "diferencia_explicada_por_centinela_usd": round(efecto_correccion, 2),
        "diferencia_sin_explicar_usd": round(
            (ingreso_post_limpieza - ingreso_bruto_crudo) - efecto_correccion, 2
        ),
    }
