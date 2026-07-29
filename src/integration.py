"""
integration.py
---------------
Fase 2: Integración y Feature Engineering.

Construye la "Sola Fuente de Verdad" a partir de los tres datasets ya
limpios (ver cleaning.py) y calcula las variables derivadas exigidas
por el challenge.

Dilema del SKU Fantasma
------------------------
Se decide tratar las ventas con SKU_ID fuera del maestro de inventario
como PRODUCTOS NUEVOS NO CATALOGADOS (falla de sincronización entre el
ERP de inventario y el sistema de ventas), NO como errores de
digitación. Evidencia que sustenta la decisión:

  1. Rango de IDs disjunto y contiguo: el inventario cubre PROD-1000 a
     PROD-3499 (2,500 SKUs); los huérfanos caen en PROD-3500 a
     PROD-4000+ — un bloque nuevo y consecutivo, no un ruido disperso
     alrededor de IDs válidos como esperaríamos de un typo.
  2. Formato 100% válido: los 1,751 SKU huérfanos cumplen exactamente
     el patrón PROD-####, sin caracteres extraños ni longitudes
     irregulares.
  3. Frecuencia de venta comparable: los SKU huérfanos se venden con
     una distribución de frecuencia (media 3.65 ventas/SKU) casi
     idéntica a la de los SKU catalogados (media 3.42 ventas/SKU) —
     el patrón de un producto real que rota en el mercado, no de un
     error aislado de tipeo.

Consecuencia para el margen: como no existe Costo_Unitario_USD para
estos productos, NO se puede calcular su margen de utilidad real.
Se cuantifica su impacto en INGRESOS (Precio_Venta_Final x Cantidad)
y se excluye explícitamente del cálculo de MARGEN AGREGADO, dejándolo
reportado por separado como "Ingreso en Riesgo por Falta de Catálogo"
en vez de forzar un supuesto de costo que inflaría o desinflaría el
margen real de la compañía.

Feedback con Transaccion_ID colisionado
-----------------------------------------
Aparte del SKU fantasma, se detectó que 767 Transaccion_ID en el
archivo de feedback tienen más de una fila asociada (1,644 filas en
total). Se comprobó que NO es un evento de negocio real (ej. cliente
que responde la encuesta dos veces): en 99% de esos grupos la
Edad_Cliente varía entre filas del mismo Transaccion_ID, lo cual es
imposible si fuera el mismo comprador. Es una falla de integridad de
llave en el generador de datos, análoga a la colisión de Feedback_ID
pero independiente de ella (solo 15% de solapamiento).

Tratamiento: en cleaning.py se marca cada fila con la bandera
'Feedback_Confiable'. Aquí, al agregar feedback por Transaccion_ID
antes del merge, se propaga esa bandera con `min` -- si CUALQUIER
fila del grupo es conflictiva, el agregado completo de esa venta
queda marcado como Feedback_Confiable = False, porque el promedio de
ratings/NPS en ese caso mezcla opiniones de clientes distintos bajo
el mismo ID. No se descarta el dato (seguiría siendo información real
de algún cliente), pero el dashboard debe poder filtrar o advertir
sobre estas ventas antes de usarlas para conclusiones de fidelidad.
"""

import pandas as pd
import numpy as np


SKU_SIN_CATALOGO = "SIN-CATALOGO"


class IntegrationError(Exception):
    """Error de negocio al integrar los tres datasets en la Sola Fuente de Verdad."""


def build_single_source_of_truth(inv: pd.DataFrame, tr: pd.DataFrame, fb: pd.DataFrame) -> dict:
    """
    Devuelve un dict con:
      - 'ventas': tabla transaccional enriquecida (transacciones + inventario
        + feedback agregado)
      - 'ratio_soporte_categoria': tabla agregada de tickets por categoría
      - 'resumen_sku_fantasma': métricas del dilema del SKU fantasma
    """
    try:
        return _build_single_source_of_truth(inv, tr, fb)
    except KeyError as exc:
        raise IntegrationError(
            f"Falta una columna esperada al integrar los datasets: {exc}. "
            f"Verifica que clean_inventario/clean_transacciones/clean_feedback "
            f"se hayan ejecutado antes de integrar."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise IntegrationError(
            f"Fallo inesperado construyendo la Sola Fuente de Verdad: {exc}"
        ) from exc


def _build_single_source_of_truth(inv: pd.DataFrame, tr: pd.DataFrame, fb: pd.DataFrame) -> dict:
    tr = tr.copy()

    # --- 1. Flag de SKU fantasma ANTES del merge (para no perder el rastro) ---
    tr["Es_SKU_Fantasma"] = ~tr["SKU_ID"].isin(inv["SKU_ID"])

    # --- 2. Left Join estratégico: transacciones es la tabla ancla ---
    #     (no se pierde ni una venta, ver docstring del módulo)
    ventas = tr.merge(inv, on="SKU_ID", how="left", suffixes=("", "_inv"))

    # Para SKU fantasma, homogenizar la categoría a algo visible en filtros
    ventas["Categoria"] = ventas["Categoria"].fillna("Sin Catálogo (SKU Fantasma)")
    ventas["Bodega_Origen"] = ventas["Bodega_Origen"].fillna("Desconocida")

    # --- 3. Enriquecer con feedback agregado por Transaccion_ID ---
    #     (una venta puede o no tener feedback asociado)
    fb_agg = fb.groupby("Transaccion_ID").agg(
        Rating_Producto=("Rating_Producto", "mean"),
        Rating_Logistica=("Rating_Logistica", "mean"),
        Satisfaccion_NPS=("Satisfaccion_NPS", "mean"),
        Ticket_Soporte_Abierto=("Ticket_Soporte_Abierto", "max"),
        Tiene_Feedback=("Feedback_ID", "count"),
        # min: si CUALQUIER fila del grupo viene de una colisión de Transaccion_ID,
        # todo el agregado de esa transacción queda marcado como no confiable —
        # el promedio de ratings/NPS en ese caso mezcla respuestas de clientes distintos.
        Feedback_Confiable=("Feedback_Confiable", "min"),
    ).reset_index()
    fb_agg["Tiene_Feedback"] = fb_agg["Tiene_Feedback"] > 0

    ventas = ventas.merge(fb_agg, on="Transaccion_ID", how="left")
    ventas["Tiene_Feedback"] = ventas["Tiene_Feedback"].fillna(False)
    ventas["Ticket_Soporte_Abierto"] = ventas["Ticket_Soporte_Abierto"].fillna(False)
    # Sin feedback asociado no es lo mismo que feedback "no confiable"; se deja explícito.
    ventas["Feedback_Confiable"] = np.where(
        ventas["Tiene_Feedback"], ventas["Feedback_Confiable"], np.nan
    )

    # =====================================================================
    # VARIABLES DERIVADAS (mínimo 3 exigidas por el challenge)
    # =====================================================================

    # --- Derivada 1: Margen de Utilidad (USD y %) ---
    # Solo calculable donde existe Costo_Unitario_USD (i.e., SKU catalogado)
    ventas["Ingreso_Total"] = ventas["Precio_Venta_Final"] * ventas["Cantidad_Vendida"]
    ventas["Costo_Total"] = np.where(
        ventas["Es_SKU_Fantasma"],
        np.nan,  # no se supone costo para productos sin catálogo
        ventas["Costo_Unitario_USD"] * ventas["Cantidad_Vendida"] + ventas["Costo_Envio"]
    )
    ventas["Margen_Utilidad_USD"] = ventas["Ingreso_Total"] - ventas["Costo_Total"]
    ventas["Margen_Utilidad_Pct"] = np.where(
        ventas["Ingreso_Total"] > 0,
        ventas["Margen_Utilidad_USD"] / ventas["Ingreso_Total"] * 100,
        np.nan
    )

    # --- Derivada 2: Brecha de Entrega vs Prometido ---
    # "Promesa" = punto medio del Lead_Time_Categoria en días (mapa de negocio)
    mapa_promesa_dias = {
        "Inmediato": 1,
        "Corto (3-5 días)": 4,
        "Medio (10 días)": 10,
        "Largo (25-30 días)": 27.5,
    }
    ventas["Dias_Prometidos"] = ventas["Lead_Time_Categoria"].map(mapa_promesa_dias)
    ventas["Brecha_Entrega_Dias"] = ventas["Tiempo_Entrega_Real"] - ventas["Dias_Prometidos"]
    # Solo tiene sentido para SKU catalogado (fantasma no tiene promesa de inventario)
    ventas.loc[ventas["Es_SKU_Fantasma"], ["Dias_Prometidos", "Brecha_Entrega_Dias"]] = np.nan

    # --- Derivada 3: Ratio de Soporte por Categoría (se calcula a nivel agregado, ver abajo) ---
    # A nivel de fila dejamos el ticket ya limpio; el ratio se agrega por categoría.

    # --- Derivada 4 (bono): Clasificación NPS individual (Promotor/Pasivo/Detractor) ---
    def clasifica_nps(x):
        if pd.isna(x):
            return "Sin Feedback"
        if x >= 50:
            return "Promotor"
        if x >= 0:
            return "Pasivo"
        return "Detractor"
    ventas["Segmento_NPS"] = ventas["Satisfaccion_NPS"].apply(clasifica_nps)

    # =====================================================================
    # Ratio de Soporte por Categoría (tabla agregada)
    # =====================================================================
    ratio_soporte = ventas.groupby("Categoria").agg(
        Total_Ventas=("Transaccion_ID", "count"),
        Tickets_Abiertos=("Ticket_Soporte_Abierto", "sum"),
    ).reset_index()
    ratio_soporte["Ratio_Soporte_Pct"] = (
        ratio_soporte["Tickets_Abiertos"] / ratio_soporte["Total_Ventas"] * 100
    ).round(2)

    # =====================================================================
    # Resumen del dilema del SKU fantasma
    # =====================================================================
    n_fantasma = ventas["Es_SKU_Fantasma"].sum()
    ingreso_fantasma = ventas.loc[ventas["Es_SKU_Fantasma"], "Ingreso_Total"].sum()
    ingreso_total = ventas["Ingreso_Total"].sum()
    resumen_sku_fantasma = {
        "n_transacciones_fantasma": int(n_fantasma),
        "pct_transacciones_fantasma": round(n_fantasma / len(ventas) * 100, 2),
        "n_sku_fantasma_unicos": int(
            ventas.loc[ventas["Es_SKU_Fantasma"], "SKU_ID"].nunique()
        ),
        "ingreso_fantasma_usd": round(ingreso_fantasma, 2),
        "ingreso_total_usd": round(ingreso_total, 2),
        "pct_ingreso_en_riesgo": round(ingreso_fantasma / ingreso_total * 100, 2),
        "margen_calculable_pct_filas": round(
            (~ventas["Margen_Utilidad_USD"].isna()).mean() * 100, 2
        ),
    }

    return {
        "ventas": ventas,
        "ratio_soporte_categoria": ratio_soporte,
        "resumen_sku_fantasma": resumen_sku_fantasma,
    }
