"""
analysis.py
-----------
Las 5 Preguntas de Alta Gerencia del challenge, como funciones puras
reutilizables tanto por app.py (dashboard) como por el documento de
hallazgos. Cada funcion recibe el dataframe `ventas` (ya filtrado por
el usuario en el sidebar) y devuelve (stats: dict, fig: matplotlib.Figure).

Todas las funciones son defensivas ante muestras pequeñas (producto de
filtros muy restrictivos): si no hay suficientes datos para un calculo
(p. ej. una correlacion), se marca explicitamente como None en vez de
arrojar una cifra poco confiable.

Ver `Documento_Hallazgos_5_Preguntas.md` para el analisis completo
sobre el dataset sin filtrar, con las pruebas estadisticas formales
(ANOVA, chi-cuadrado) que sustentan cada conclusion.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - Agg debe fijarse antes de este import

COLOR_MAIN = "#2563eb"
COLOR_NEG = "#dc2626"
MIN_N_CORR = 30  # tamaño mínimo de muestra para reportar una correlación


class AnalysisError(Exception):
    """Error de negocio calculando una de las 5 preguntas estratégicas."""


def _empty_fig(msg="No hay suficientes datos para este cálculo con los filtros actuales."):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def _safe(func):
    """Decorador: envuelve errores inesperados en AnalysisError con contexto."""

    def wrapper(df, *args, **kwargs):
        try:
            return func(df, *args, **kwargs)
        except KeyError as exc:
            raise AnalysisError(
                f"{func.__name__}: falta la columna {exc} en el dataframe. "
                f"¿Se construyó con build_single_source_of_truth?"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"{func.__name__}: fallo inesperado: {exc}") from exc

    wrapper.__name__ = func.__name__
    return wrapper


# ---------------------------------------------------------------------------
# Q1 — Fuga de Capital y Rentabilidad
# ---------------------------------------------------------------------------
@_safe
def q1_margen_canal(df: pd.DataFrame):
    cat_df = df.dropna(subset=["Margen_Utilidad_USD"])
    if cat_df.empty:
        return {"n": 0}, _empty_fig()

    perdida = cat_df.loc[cat_df["Margen_Utilidad_USD"] < 0, "Margen_Utilidad_USD"].sum()
    n_neg = int((cat_df["Margen_Utilidad_USD"] < 0).sum())
    corr_precio_costo = cat_df["Precio_Venta_Final"].corr(cat_df["Costo_Unitario_USD"])

    stats = {
        "n_ventas_catalogadas": int(len(cat_df)),
        "n_margen_negativo": n_neg,
        "pct_margen_negativo": round(n_neg / len(cat_df) * 100, 2),
        "perdida_total_usd": round(float(perdida), 2),
        "margen_neto_usd": round(float(cat_df["Margen_Utilidad_USD"].sum()), 2),
        "corr_precio_costo": round(float(corr_precio_costo), 4),
    }
    canal_stats = (
        cat_df.groupby("Canal_Venta")["Margen_Utilidad_USD"]
        .apply(lambda x: (x < 0).mean() * 100)
        .sort_values()
    )
    stats["pct_margen_negativo_por_canal"] = canal_stats.round(2).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(canal_stats.index, canal_stats.values, color=COLOR_MAIN)
    ax.set_xlabel("% de ventas con margen negativo")
    ax.set_title("% Margen Negativo por Canal de Venta")
    for i, v in enumerate(canal_stats.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    fig.tight_layout()
    return stats, fig


# ---------------------------------------------------------------------------
# Q2 — Crisis Logística y Cuellos de Botella
# ---------------------------------------------------------------------------
@_safe
def q2_logistica_nps(df: pd.DataFrame):
    d = df.dropna(subset=["Satisfaccion_NPS", "Tiempo_Entrega_Real"])
    if len(d) < MIN_N_CORR:
        stats = {"n": len(d), "nota": "Muestra insuficiente para correlación confiable."}
        return stats, _empty_fig()

    corr_global = d["Tiempo_Entrega_Real"].corr(d["Satisfaccion_NPS"])
    stats = {"n": int(len(d)), "corr_global_tiempo_nps": round(float(corr_global), 4)}

    problema = df["Estado_Envio"].isin(["Retrasado", "Perdido", "Devuelto"])
    tasa_bodega = (
        df.assign(problema=problema)
        .groupby("Bodega_Origen")["problema"]
        .mean()
        .mul(100)
        .sort_values()
    )
    stats["pct_envio_problematico_por_bodega"] = tasa_bodega.round(2).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(tasa_bodega.index, tasa_bodega.values, color=COLOR_MAIN)
    ax.set_xlabel("% envíos Retrasado / Perdido / Devuelto")
    ax.set_title("Tasa de Envíos Problemáticos por Bodega")
    fig.tight_layout()
    return stats, fig


# ---------------------------------------------------------------------------
# Q3 — Análisis de la Venta Invisible
# ---------------------------------------------------------------------------
@_safe
def q3_sku_fantasma(df: pd.DataFrame):
    if df.empty:
        return {"n": 0}, _empty_fig()

    ingreso_total = float(df["Ingreso_Total"].sum())
    ingreso_fantasma = float(df.loc[df["Es_SKU_Fantasma"], "Ingreso_Total"].sum())
    pct_riesgo = round(ingreso_fantasma / ingreso_total * 100, 2) if ingreso_total > 0 else None
    stats = {
        "n_transacciones_fantasma": int(df["Es_SKU_Fantasma"].sum()),
        "pct_transacciones_fantasma": round(df["Es_SKU_Fantasma"].mean() * 100, 2),
        "ingreso_total_usd": round(ingreso_total, 2),
        "ingreso_fantasma_usd": round(ingreso_fantasma, 2),
        "pct_ingreso_en_riesgo": pct_riesgo,
    }

    fig, ax = plt.subplots(figsize=(6, 5))
    catalogado = ingreso_total - ingreso_fantasma
    if ingreso_total > 0:
        ax.pie(
            [catalogado, ingreso_fantasma],
            labels=[
                f"Catalogado\n${catalogado / 1e6:.1f}M",
                f"SKU Fantasma\n${ingreso_fantasma / 1e6:.1f}M",
            ],
            colors=[COLOR_MAIN, COLOR_NEG],
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
    ax.set_title("Ingreso: Catalogado vs. SKU Fantasma")
    fig.tight_layout()
    return stats, fig


# ---------------------------------------------------------------------------
# Q4 — Diagnóstico de Fidelidad
# ---------------------------------------------------------------------------
@_safe
def q4_paradoja_fidelidad(df: pd.DataFrame):
    stock_cat = df.groupby("Categoria")["Stock_Actual"].mean()
    sent_cat = (
        df.dropna(subset=["Satisfaccion_NPS"]).groupby("Categoria")["Satisfaccion_NPS"].mean()
    )
    combo = pd.DataFrame({"stock": stock_cat, "nps": sent_cat}).dropna()
    if combo.empty:
        return {"n": 0}, _empty_fig()

    mediana_stock = combo["stock"].median()
    combo["paradoja"] = (combo["stock"] > mediana_stock) & (combo["nps"] < 0)
    stats = {
        "tabla": combo.round(2).to_dict(orient="index"),
        "categorias_con_paradoja": combo[combo["paradoja"]].index.tolist(),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = [COLOR_NEG if p else COLOR_MAIN for p in combo["paradoja"]]
    ax.scatter(combo["stock"], combo["nps"], s=200, color=colors, edgecolor="black", zorder=3)
    for cat, row in combo.iterrows():
        ax.annotate(
            cat, (row["stock"], row["nps"]), xytext=(5, 5),
            textcoords="offset points", fontsize=9,
        )
    ax.axhline(0, color="#9ca3af", linestyle="--", linewidth=1)
    ax.axvline(mediana_stock, color="#9ca3af", linestyle="--", linewidth=1)
    ax.set_xlabel("Stock Promedio (unidades)")
    ax.set_ylabel("NPS Promedio")
    ax.set_title("Stock Disponible vs. Sentimiento por Categoría")
    fig.tight_layout()
    return stats, fig


# ---------------------------------------------------------------------------
# Q5 — Storytelling de Riesgo Operativo
# ---------------------------------------------------------------------------
@_safe
def q5_ceguera_inventario(df: pd.DataFrame, hoy: pd.Timestamp = None):
    if hoy is None:
        hoy = pd.Timestamp.today()

    d = df.dropna(subset=["Ultima_Revision"]).drop_duplicates(subset=["SKU_ID"]).copy()
    if d.empty:
        return {"n": 0}, _empty_fig()

    d["Antiguedad_Dias"] = (hoy - pd.to_datetime(d["Ultima_Revision"])).dt.days
    antiguedad_bodega = d.groupby("Bodega_Origen")["Antiguedad_Dias"].mean().sort_values()

    ticket_sku = (
        df[~df["Es_SKU_Fantasma"]]
        .groupby("SKU_ID")["Ticket_Soporte_Abierto"]
        .mean()
        .astype(float)
        .mul(100)
    )
    merged = (
        d.set_index("SKU_ID")
        .join(ticket_sku.rename("tasa_ticket"), how="inner")
        .reset_index()
    )

    stats = {
        "antiguedad_promedio_dias_por_bodega": antiguedad_bodega.round(1).to_dict(),
        "antiguedad_promedio_global_dias": round(float(d["Antiguedad_Dias"].mean()), 1),
    }
    if len(merged) >= MIN_N_CORR:
        corr = merged["Antiguedad_Dias"].astype(float).corr(merged["tasa_ticket"].astype(float))
        stats["corr_antiguedad_tasa_ticket"] = round(float(corr), 4)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(antiguedad_bodega.index, antiguedad_bodega.values, color=COLOR_MAIN)
    ax.axvline(365, color=COLOR_NEG, linestyle="--", label="1 año")
    ax.set_xlabel("Antigüedad promedio última revisión (días)")
    ax.set_title('"Ceguera" de Inventario por Bodega')
    ax.legend()
    fig.tight_layout()
    return stats, fig
