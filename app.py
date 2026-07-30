"""
app.py
------
TechLogistics S.A.S. -- Sistema de Soporte a la Decision (DSS)
Challenge 02 - Fundamentos en Ciencia de Datos (Maestria) - EAFIT

Ejecutar localmente:
    streamlit run app.py

La logica de negocio (limpieza, integracion, feature engineering e IA)
vive en src/*.py, separada de esta capa de interfaz, siguiendo el
requisito de codigo modular del challenge.
"""

import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from quality import (  # noqa: E402
    null_report, duplicate_report, outlier_report, health_score, revenue_reconciliation,
    outlier_report_by_group,
)
from cleaning import (  # noqa: E402
    clean_inventario, clean_transacciones, clean_feedback, DataCleaningError,
)
from integration import build_single_source_of_truth, IntegrationError  # noqa: E402
from ai_module import build_stat_summary, call_groq, GROQ_MODEL, GroqAPIError  # noqa: E402
from analysis import (  # noqa: E402
    q1_margen_canal, q2_logistica_nps, q3_sku_fantasma,
    q4_paradoja_fidelidad, q5_ceguera_inventario, AnalysisError,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(
    page_title="TechLogistics · DSS",
    page_icon="📦",
    layout="wide",
)


# =============================================================================
# CARGA Y PIPELINE (cacheados para no recalcular en cada interacción)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_raw_data():
    inv = pd.read_csv(os.path.join(DATA_DIR, "inventario_central_v2.csv"))
    tr = pd.read_csv(os.path.join(DATA_DIR, "transacciones_logistica_v2.csv"))
    fb = pd.read_csv(os.path.join(DATA_DIR, "feedback_clientes_v2.csv"))
    return inv, tr, fb


@st.cache_data(show_spinner=False)
def run_pipeline():
    inv_raw, tr_raw, fb_raw = load_raw_data()

    inv_clean, log_inv = clean_inventario(inv_raw)
    tr_clean, log_tr = clean_transacciones(tr_raw)
    fb_clean, log_fb = clean_feedback(fb_raw)

    result = build_single_source_of_truth(inv_clean, tr_clean, fb_clean)

    raw = {"Inventario": inv_raw, "Transacciones": tr_raw, "Feedback": fb_raw}
    clean = {"Inventario": inv_clean, "Transacciones": tr_clean, "Feedback": fb_clean}
    logs = {"Inventario": log_inv, "Transacciones": log_tr, "Feedback": log_fb}
    id_cols = {
        "Inventario": "SKU_ID",
        "Transacciones": "Transaccion_ID",
        "Feedback": "Feedback_ID",
    }

    return raw, clean, logs, id_cols, result


try:
    raw_data, clean_data, cleaning_logs, id_cols, integration_result = run_pipeline()
    ventas = integration_result["ventas"]
    ratio_soporte = integration_result["ratio_soporte_categoria"]
    resumen_fantasma = integration_result["resumen_sku_fantasma"]
except FileNotFoundError as exc:
    st.error(
        "No se encontraron los archivos CSV en la carpeta `data/`. "
        "Asegúrate de que el repositorio incluya `data/inventario_central_v2.csv`, "
        "`data/transacciones_logistica_v2.csv` y `data/feedback_clientes_v2.csv`.\n\n"
        f"Detalle técnico: {exc}"
    )
    st.stop()
except (DataCleaningError, IntegrationError) as exc:
    st.error(
        "Ocurrió un problema durante la limpieza o integración de los datos. "
        "Esto normalmente indica que el esquema de alguno de los CSV cambió.\n\n"
        f"Detalle técnico: {exc}"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - último recurso, nunca debe tumbar la app sin explicar
    st.error(f"Error inesperado inicializando el dashboard: {exc}")
    st.stop()


# =============================================================================
# SIDEBAR — Filtros, refresco y API Key
# =============================================================================
st.sidebar.title("📦 TechLogistics DSS")
st.sidebar.caption("Consultoría Senior · Challenge 02 · EAFIT")

if st.sidebar.button("🔄 Refrescar Análisis", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("### 🔎 Filtros")

fecha_min = ventas["Fecha_Venta"].min().date()
fecha_max = ventas["Fecha_Venta"].max().date()
rango_fechas = st.sidebar.date_input(
    "Rango de fechas de venta",
    value=(fecha_min, fecha_max),
    min_value=fecha_min,
    max_value=fecha_max,
)
if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
    fecha_inicio, fecha_fin = rango_fechas
else:
    fecha_inicio, fecha_fin = fecha_min, fecha_max

categorias_disponibles = sorted(ventas["Categoria"].dropna().unique().tolist())
categorias_sel = st.sidebar.multiselect(
    "Categoría", categorias_disponibles, default=categorias_disponibles
)

bodegas_disponibles = sorted(ventas["Bodega_Origen"].dropna().unique().tolist())
bodegas_sel = st.sidebar.multiselect(
    "Bodega de origen", bodegas_disponibles, default=bodegas_disponibles
)

ciudades_disponibles = sorted(ventas["Ciudad_Destino"].dropna().unique().tolist())
ciudades_sel = st.sidebar.multiselect(
    "Ciudad destino", ciudades_disponibles, default=ciudades_disponibles
)

canal_disponible = sorted(ventas["Canal_Venta"].dropna().unique().tolist())
canal_sel = st.sidebar.multiselect(
    "Canal de venta", canal_disponible, default=canal_disponible
)

incluir_fantasma = st.sidebar.checkbox(
    "Incluir ventas con SKU Fantasma (sin catálogo)", value=True,
    help="Si se desmarca, se excluyen las ventas cuyo SKU no existe en el inventario maestro.",
)

# --- Aplicar filtros ---
mask = (
    (ventas["Fecha_Venta"].dt.date >= fecha_inicio)
    & (ventas["Fecha_Venta"].dt.date <= fecha_fin)
    & (ventas["Categoria"].isin(categorias_sel))
    & (ventas["Bodega_Origen"].isin(bodegas_sel))
    & (ventas["Ciudad_Destino"].isin(ciudades_sel))
    & (ventas["Canal_Venta"].isin(canal_sel))
)
if not incluir_fantasma:
    mask &= ~ventas["Es_SKU_Fantasma"]

df = ventas.loc[mask].copy()

st.sidebar.markdown("---")
st.sidebar.caption(f"**{len(df):,}** transacciones seleccionadas de {len(ventas):,} totales")

# --- Módulo IA: API Key ---
st.sidebar.markdown("### 🤖 Groq API Key")
groq_api_key = st.sidebar.text_input(
    "Ingresa tu Groq API Key",
    type="password",
    placeholder="gsk_...",
    help=(
        "Necesaria para generar las recomendaciones de IA en la pestaña 'Insights IA'. "
        "Consíguela gratis en https://console.groq.com/keys. "
        "No se guarda en ningún lado: solo vive en esta sesión del navegador y se envía "
        "directamente a la API de Groq."
    ),
)
st.sidebar.caption(
    "🔒 La API Key no se almacena ni se registra en ningún log de la aplicación."
)


# =============================================================================
# TABS PRINCIPALES
# =============================================================================
TAB_LABELS = [
    "🔍 Auditoría de Calidad",
    "📦 Operaciones",
    "👥 Cliente",
    "📋 5 Preguntas Estratégicas",
    "🤖 Insights de IA",
]
tab_auditoria, tab_operaciones, tab_cliente, tab_preguntas, tab_ia = st.tabs(TAB_LABELS)

# -----------------------------------------------------------------------
# TAB 1 — AUDITORÍA (Fase 1: transparencia, antes vs después)
# -----------------------------------------------------------------------
with tab_auditoria:
    st.header("Módulo de Transparencia: Antes vs. Después")
    st.caption(
        "Un consultor senior no limpia datos sin dejar rastro. Aquí se documenta cada "
        "corrección aplicada y su justificación."
    )

    dataset_sel = st.selectbox("Selecciona el dataset a auditar", list(raw_data.keys()))
    df_raw = raw_data[dataset_sel]
    df_clean = clean_data[dataset_sel]
    id_col = id_cols[dataset_sel]

    numeric_cols_map = {
        "Inventario": ["Stock_Actual", "Costo_Unitario_USD", "Punto_Reorden"],
        "Transacciones": [
            "Cantidad_Vendida", "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real",
        ],
        "Feedback": ["Rating_Producto", "Rating_Logistica", "Edad_Cliente", "Satisfaccion_NPS"],
    }
    numeric_cols = [c for c in numeric_cols_map[dataset_sel] if c in df_raw.columns]

    try:
        hs_antes = health_score(df_raw, numeric_cols, id_col)
        numeric_cols_clean = [c for c in numeric_cols if c in df_clean.columns]
        hs_despues = health_score(df_clean, numeric_cols_clean, id_col)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo calcular el Health Score de '{dataset_sel}': {exc}")
        hs_antes = {"health_score": 0.0}
        hs_despues = {"health_score": 0.0}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health Score (antes)", f"{hs_antes['health_score']:.1f} / 100")
    c2.metric(
        "Health Score (después)", f"{hs_despues['health_score']:.1f} / 100",
        delta=f"{hs_despues['health_score'] - hs_antes['health_score']:.1f}",
    )
    filas_eliminadas = len(df_raw) - len(df_clean)
    c3.metric(
        "Filas eliminadas", f"{filas_eliminadas:,}",
        help=(
            "Política del proyecto: nunca se elimina una fila por un solo campo sucio; "
            "se imputa o se etiqueta explícitamente (ver Fase 1). Por eso este número "
            "es 0 en los tres datasets — es una decisión documentada, no un descuido."
        ),
    )
    c4.metric("Filas totales (post-limpieza)", f"{len(df_clean):,}")

    st.subheader("Bitácora de limpieza aplicada")
    for accion in cleaning_logs[dataset_sel]["acciones"]:
        st.markdown(f"- {accion}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Nulidad por columna — ANTES")
        st.dataframe(null_report(df_raw), use_container_width=True, hide_index=True)
    with col_b:
        st.subheader("Nulidad por columna — DESPUÉS")
        st.dataframe(null_report(df_clean), use_container_width=True, hide_index=True)

    st.subheader("Duplicados detectados (sobre datos crudos)")
    dupes = duplicate_report(df_raw, id_col)
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Filas 100% idénticas", dupes.get("full_row_duplicates", 0))
    dc2.metric(f"Filas con {id_col} colisionado", dupes.get("id_collisions_rows", 0))
    dc3.metric(f"{id_col} únicos afectados", dupes.get("id_collisions_unique_ids", 0))

    st.subheader("Outliers detectados (IQR, sobre datos crudos)")
    try:
        st.dataframe(
            outlier_report(df_raw, numeric_cols), use_container_width=True, hide_index=True
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo calcular la tabla de outliers: {exc}")
    st.caption(
        "⚠️ Este IQR es GLOBAL (una sola distribución para toda la columna). Un valor "
        "puede ser normal para una categoría y anómalo para otra sin que este cálculo "
        "lo distinga — ver el desglose por categoría abajo, metodológicamente más "
        "riguroso."
    )

    # Columnas candidatas para agrupar el IQR de forma más rigurosa que el
    # global -- distintas por dataset, según qué dimensiones categóricas
    # tienen sentido de negocio en cada uno.
    grupo_cols_map = {
        "Inventario": ["Categoria", "Bodega_Origen"],
        "Transacciones": ["Ciudad_Destino", "Canal_Venta"],
        "Feedback": [],  # sin dimensión categórica de negocio relevante para IQR
    }
    grupo_cols_disponibles = [c for c in grupo_cols_map[dataset_sel] if c in df_raw.columns]

    if grupo_cols_disponibles:
        with st.expander("🔬 Outliers IQR calculados por grupo (más riguroso que el global)"):
            st.caption(
                "El mismo método IQR (1.5×), pero calculado de forma independiente "
                "dentro de cada grupo en vez de sobre toda la columna a la vez -- un "
                "valor puede ser normal para un grupo y anómalo para otro sin que el "
                "IQR global lo distinga."
            )
            gc1, gc2 = st.columns(2)
            with gc1:
                col_iqr_grupo = st.selectbox(
                    "Columna numérica a analizar", numeric_cols, key="iqr_by_group_col"
                )
            with gc2:
                agrupar_por = st.selectbox(
                    "Agrupar por", grupo_cols_disponibles, key="iqr_by_group_dim"
                )
            try:
                st.dataframe(
                    outlier_report_by_group(df_raw, col_iqr_grupo, agrupar_por),
                    use_container_width=True, hide_index=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"No se pudo calcular el desglose por {agrupar_por}: {exc}")
            st.caption(
                "En este dataset, el desglose por grupo no cambia sustancialmente los "
                "outliers detectados en ninguna de las columnas numéricas de "
                "Inventario ni de Transacciones: las distribuciones (mediana, Q1, Q3) "
                "son casi idénticas entre categorías, ciudades y canales — consistente "
                "con el hallazgo de la Pregunta 2 (Fase Analítica) de que no hay una "
                "zona logística estadísticamente distinta de las demás. Esta vista "
                "queda disponible para auditar ese supuesto en cualquier filtro futuro, "
                "en vez de darlo por hecho."
            )

    registros_excluidos = cleaning_logs[dataset_sel].get("registros_excluidos")
    if registros_excluidos is not None and not registros_excluidos.empty:
        with st.expander(
            f"🔎 Ver registros excluidos/winsorizados ({len(registros_excluidos)})"
        ):
            st.caption(
                "Estos registros NO se eliminaron: su valor original se capó a los "
                "percentiles P1/P99 para evitar que distorsionen los KPIs agregados, "
                "pero el producto y su venta siguen presentes en el análisis."
            )
            st.dataframe(registros_excluidos, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar reporte de limpieza (JSON)",
        data=json.dumps(cleaning_logs, ensure_ascii=False, indent=2, default=str),
        file_name="reporte_limpieza_techlogistics.json",
        mime="application/json",
    )

    st.markdown("---")
    st.subheader("📅 Validación Temporal (fechas futuras)")
    n_fechas_futuras = int(clean_data["Transacciones"]["Fecha_Venta_Futura"].sum())
    if n_fechas_futuras > 0:
        st.warning(
            f"⚠️ {n_fechas_futuras} transacciones tienen una fecha posterior al momento "
            f"de ejecución de este análisis. Se excluyen de los gráficos de series de "
            f"tiempo pero se conservan en la tabla para auditoría."
        )
    else:
        st.success(
            "✅ Validación dinámica contra la fecha actual (no una fecha fija en el "
            "código): no se encontraron transacciones con fecha futura en esta ejecución."
        )

    st.markdown("---")
    st.subheader("💰 Trazabilidad de Ingresos (crudo → limpio)")
    reconciliacion = revenue_reconciliation(
        raw_data["Transacciones"], clean_data["Transacciones"]
    )
    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Ingreso bruto (archivo crudo)", f"${reconciliacion['ingreso_bruto_crudo_usd']:,.0f}"
    )
    r2.metric(
        "Ingreso post-limpieza", f"${reconciliacion['ingreso_post_limpieza_usd']:,.0f}",
        delta=f"${reconciliacion['diferencia_usd']:,.0f}",
    )
    r3.metric(
        "Diferencia sin explicar", f"${reconciliacion['diferencia_sin_explicar_usd']:,.2f}",
        help="Debe ser ≈$0: toda diferencia debe quedar explicada por una corrección "
             "documentada.",
    )
    st.caption(
        f"La diferencia total se explica al 100% por la corrección del centinela "
        f"Cantidad_Vendida = -5 en {reconciliacion['n_filas_centinela_cantidad']} filas "
        f"(Fase 1): ${reconciliacion['diferencia_explicada_por_centinela_usd']:,.2f}. "
        f"Este es el requisito de trazabilidad de la Guía de Validación — el ingreso "
        f"final es reconciliable, no una caja negra."
    )

    st.markdown("---")
    st.subheader("🕵️ Dilema del SKU Fantasma")
    f1, f2, f3 = st.columns(3)
    f1.metric(
        "Ventas con SKU sin catálogo",
        f"{resumen_fantasma['n_transacciones_fantasma']:,}",
        f"{resumen_fantasma['pct_transacciones_fantasma']}% del total",
    )
    f2.metric("Ingreso en riesgo (USD)", f"${resumen_fantasma['ingreso_fantasma_usd']:,.0f}")
    f3.metric("% del ingreso total en riesgo", f"{resumen_fantasma['pct_ingreso_en_riesgo']}%")
    st.info(
        "Decisión: se tratan como **productos nuevos no catalogados** (falla de "
        "sincronización ERP↔Ventas), no como errores de digitación — evidencia: rango de "
        "SKU contiguo y separado, formato 100% válido, y frecuencia de venta comparable a "
        "los productos catalogados. Ver Fase 2 para el detalle completo."
    )


# -----------------------------------------------------------------------
# TAB 2 — OPERACIONES (margen, logística)
# -----------------------------------------------------------------------
with tab_operaciones:
    st.header("Rentabilidad y Operación Logística")

    if df.empty:
        st.warning("No hay transacciones para los filtros seleccionados.")
    else:
        ingreso_total = df["Ingreso_Total"].sum()
        margen_total = df["Margen_Utilidad_USD"].sum(skipna=True)
        margen_valido = df["Margen_Utilidad_USD"].dropna()
        pct_margen_neg = (margen_valido < 0).mean() * 100 if len(margen_valido) else None
        brecha_prom = df["Brecha_Entrega_Dias"].mean(skipna=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Ingreso total (filtrado)", f"${ingreso_total:,.0f}")
        m2.metric("Margen total (solo SKU catalogado)", f"${margen_total:,.0f}")
        m3.metric(
            "% ventas con margen negativo",
            f"{pct_margen_neg:.1f}%" if pct_margen_neg is not None else "N/A",
        )
        m4.metric(
            "Brecha de entrega promedio",
            f"{brecha_prom:+.1f} días" if pd.notna(brecha_prom) else "N/A",
        )

        st.subheader("Margen de utilidad por categoría")
        margen_cat = df.groupby("Categoria")["Margen_Utilidad_USD"].sum(numeric_only=True)
        st.bar_chart(margen_cat.sort_values())

        st.subheader("Ingreso por ciudad de destino")
        ingreso_ciudad = df.groupby("Ciudad_Destino")["Ingreso_Total"].sum()
        st.bar_chart(ingreso_ciudad.sort_values(ascending=False))

        st.subheader("Distribución de la brecha de entrega (días vs. promesa)")
        st.caption(
            "Positivo = se entregó más tarde de lo prometido por el Lead Time del proveedor."
        )
        brecha_dist = df["Brecha_Entrega_Dias"].dropna().round(0).value_counts().sort_index()
        st.bar_chart(brecha_dist)

        st.subheader("Ratio de tickets de soporte por categoría")
        st.dataframe(ratio_soporte, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------
# TAB 3 — CLIENTE (NPS, fidelidad)
# -----------------------------------------------------------------------
with tab_cliente:
    st.header("Voz del Cliente y Fidelidad")

    if df.empty:
        st.warning("No hay transacciones para los filtros seleccionados.")
    else:
        con_fb = df[df["Tiene_Feedback"]]
        c1, c2, c3 = st.columns(3)
        pct_con_fb = len(con_fb) / len(df) * 100 if len(df) else 0
        c1.metric("Ventas con feedback", f"{len(con_fb):,} ({pct_con_fb:.1f}%)")

        no_confiable = 0
        if "Feedback_Confiable" in con_fb.columns:
            no_confiable = int(con_fb["Feedback_Confiable"].eq(False).sum())
        c2.metric(
            "Feedback potencialmente no confiable",
            f"{no_confiable:,}",
            help="Ventas cuyo Transaccion_ID estaba colisionado en el archivo original.",
        )
        c3.metric("Tickets de soporte abiertos", f"{int(df['Ticket_Soporte_Abierto'].sum()):,}")

        st.subheader("Segmentación NPS")
        st.bar_chart(df["Segmento_NPS"].value_counts())

        st.subheader("Rating promedio de producto por categoría")
        rating_cat = df.groupby("Categoria")["Rating_Producto"].mean(numeric_only=True)
        st.bar_chart(rating_cat.sort_values())

        if no_confiable > 0:
            st.warning(
                f"⚠️ {no_confiable} ventas del filtro actual tienen feedback marcado como no "
                "confiable (Transaccion_ID colisionado en el archivo original — ver Fase 1). "
                "Interprete el NPS y los ratings de esas ventas con cautela."
            )


# -----------------------------------------------------------------------
# TAB 4 — 5 PREGUNTAS ESTRATÉGICAS
# -----------------------------------------------------------------------
with tab_preguntas:
    st.header("📋 Las 5 Preguntas de Alta Gerencia")
    st.caption(
        "Todo lo calculado aquí usa exactamente los filtros aplicados en la barra lateral. "
        "Con muestras pequeñas algunos cálculos (correlaciones) se omiten para no reportar "
        "cifras poco confiables. Ver el Documento de Hallazgos para el análisis completo con "
        "pruebas estadísticas formales (ANOVA, chi-cuadrado) sobre el dataset sin filtrar."
    )

    if df.empty:
        st.warning("No hay transacciones para los filtros seleccionados.")
    else:
        # ---- Q1 ----
        st.subheader("1️⃣ Fuga de Capital y Rentabilidad")
        try:
            s1, fig1 = q1_margen_canal(df)
        except AnalysisError as exc:
            st.error(f"No se pudo calcular la Pregunta 1: {exc}")
            s1, fig1 = {}, None

        if s1.get("n_ventas_catalogadas", 0) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric("% ventas con margen negativo", f"{s1['pct_margen_negativo']}%")
            c2.metric("Pérdida total (USD)", f"${s1['perdida_total_usd']:,.0f}")
            c3.metric(
                "Corr. Precio vs. Costo", f"{s1['corr_precio_costo']}",
                help="≈0 = el precio no está ligado al costo real",
            )
            st.pyplot(fig1)
            st.markdown(
                "**Diagnóstico:** la correlación entre precio y costo es prácticamente "
                "nula — no es una falla del canal Online (de hecho es el canal con "
                "*menor* proporción de ventas en pérdida), es una falla sistémica de "
                "gobernanza de precios."
            )
        elif fig1 is not None:
            st.info("No hay ventas con SKU catalogado en el filtro actual.")

        st.markdown("---")

        # ---- Q2 ----
        st.subheader("2️⃣ Crisis Logística y Cuellos de Botella")
        try:
            s2, fig2 = q2_logistica_nps(df)
        except AnalysisError as exc:
            st.error(f"No se pudo calcular la Pregunta 2: {exc}")
            s2, fig2 = {}, None

        if "corr_global_tiempo_nps" in s2:
            c1, c2 = st.columns(2)
            c1.metric(
                "Correlación Tiempo Entrega ↔ NPS", f"{s2['corr_global_tiempo_nps']}",
                help="Cercano a 0 = sin relación lineal",
            )
            tasas_bodega = s2["pct_envio_problematico_por_bodega"]
            peor_bodega = max(tasas_bodega, key=tasas_bodega.get)
            c2.metric(
                "Bodega con más envíos problemáticos", peor_bodega,
                f"{tasas_bodega[peor_bodega]:.1f}%",
            )
            st.pyplot(fig2)
            st.markdown(
                "**Diagnóstico:** la correlación entre tiempo de entrega y NPS es "
                "estadísticamente insignificante en todos los niveles de agregación "
                "probados. El problema real no es una zona específica: es que **toda "
                "la red logística** tiene una tasa de envíos problemáticos cercana al "
                "50%, de forma casi uniforme."
            )
        elif fig2 is not None:
            st.info(s2.get("nota", "Muestra insuficiente para este cálculo."))

        st.markdown("---")

        # ---- Q3 ----
        st.subheader("3️⃣ Análisis de la Venta Invisible")
        try:
            s3, fig3 = q3_sku_fantasma(df)
        except AnalysisError as exc:
            st.error(f"No se pudo calcular la Pregunta 3: {exc}")
            s3, fig3 = {}, None

        if s3.get("ingreso_total_usd", 0) > 0:
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Transacciones SKU fantasma", f"{s3['n_transacciones_fantasma']:,}",
                f"{s3['pct_transacciones_fantasma']}%",
            )
            c2.metric("Ingreso en riesgo (USD)", f"${s3['ingreso_fantasma_usd']:,.0f}")
            c3.metric("% del ingreso total en riesgo", f"{s3['pct_ingreso_en_riesgo']}%")
            st.pyplot(fig3)
            st.markdown(
                "**Diagnóstico:** son productos nuevos no catalogados (falla ERP↔Ventas), "
                "no errores de digitación — ver Fase 2 para la evidencia completa. Sin "
                "costo verificado, no se puede saber si este ingreso es rentable."
            )
        elif fig3 is not None:
            st.info("No hay ingreso en el filtro actual.")

        st.markdown("---")

        # ---- Q4 ----
        st.subheader("4️⃣ Diagnóstico de Fidelidad")
        try:
            s4, fig4 = q4_paradoja_fidelidad(df)
        except AnalysisError as exc:
            st.error(f"No se pudo calcular la Pregunta 4: {exc}")
            s4, fig4 = {}, None

        if s4.get("tabla"):
            paradoja = s4["categorias_con_paradoja"]
            if paradoja:
                st.warning(
                    f"Categoría(s) con stock alto y NPS negativo: **{', '.join(paradoja)}**"
                )
            else:
                st.success(
                    "No se detectan categorías con la paradoja stock-alto/sentimiento-"
                    "negativo en este filtro."
                )
            st.pyplot(fig4)
            st.markdown(
                "**Diagnóstico (dataset completo):** la diferencia de NPS y de rating "
                "entre categorías no es estadísticamente significativa (ANOVA p=0.20 y "
                "p=0.97). Ni 'mala calidad' ni 'sobrecosto' explican la paradoja — "
                "probablemente sea ruido por baja cobertura de feedback."
            )
        elif fig4 is not None:
            st.info("No hay suficientes categorías con feedback en el filtro actual.")

        st.markdown("---")

        # ---- Q5 ----
        st.subheader("5️⃣ Storytelling de Riesgo Operativo")
        try:
            s5, fig5 = q5_ceguera_inventario(df)
        except AnalysisError as exc:
            st.error(f"No se pudo calcular la Pregunta 5: {exc}")
            s5, fig5 = {}, None

        if "antiguedad_promedio_global_dias" in s5:
            c1, c2 = st.columns(2)
            c1.metric(
                "Antigüedad promedio de revisión (todas las bodegas)",
                f"{s5['antiguedad_promedio_global_dias']:.0f} días",
            )
            if "corr_antiguedad_tasa_ticket" in s5:
                c2.metric(
                    "Correlación antigüedad ↔ tickets",
                    f"{s5['corr_antiguedad_tasa_ticket']}",
                )
            st.pyplot(fig5)
            st.markdown(
                "**Diagnóstico:** no hay correlación con la tasa de tickets, pero el "
                "hallazgo real es el nivel absoluto: **todas las bodegas llevan más de "
                "un año sin un recuento físico**, sin excepción. Es la causa más "
                "probable de los 60 registros de stock negativo detectados en la Fase 1."
            )
        elif fig5 is not None:
            st.info("No hay suficientes datos de inventario en el filtro actual.")


# -----------------------------------------------------------------------
# TAB 5 — INSIGHTS DE IA (Fase 3: Groq / Llama 3.3 70B)
# -----------------------------------------------------------------------
with tab_ia:
    st.header("🤖 Recomendaciones Estratégicas con IA (Llama 3.3 70B vía Groq)")
    st.caption(
        "El modelo analiza el resumen estadístico de los datos que tienes filtrados "
        "en la barra lateral en este momento — no ve el dataset completo, solo el "
        "subconjunto seleccionado."
    )

    if df.empty:
        st.warning(
            "No hay transacciones para los filtros seleccionados. "
            "Ajusta los filtros para generar el resumen."
        )
    else:
        try:
            resumen = build_stat_summary(df)
        except KeyError as exc:
            st.error(f"No se pudo construir el resumen estadístico: {exc}")
            resumen = None

        if resumen is not None:
            with st.expander("📊 Ver el resumen estadístico que se enviará al modelo"):
                st.json(resumen)

            generar = st.button("✨ Generar Recomendaciones Estratégicas", type="primary")

            if generar:
                if not groq_api_key:
                    st.error(
                        "Debes ingresar tu Groq API Key en la barra lateral (sección "
                        "'🤖 Groq API Key') antes de generar recomendaciones."
                    )
                else:
                    with st.spinner(f"Consultando {GROQ_MODEL} en Groq..."):
                        try:
                            texto = call_groq(groq_api_key, resumen, model=GROQ_MODEL)
                        except GroqAPIError as exc:
                            st.error(str(exc))
                        except Exception as exc:  # noqa: BLE001 - último recurso
                            st.error(f"Error inesperado llamando a Groq: {exc}")
                        else:
                            st.success("Recomendaciones generadas:")
                            st.markdown(texto)
                            st.download_button(
                                "⬇️ Descargar recomendaciones (TXT)",
                                data=texto,
                                file_name="recomendaciones_ia_techlogistics.txt",
                                mime="text/plain",
                            )

    st.markdown("---")
    st.caption(
        "ℹ️ Nota técnica: Groq anunció el retiro del modelo `llama-3.3-70b-versatile` "
        "para el 16-ago-2026, recomendando migrar a `openai/gpt-oss-120b`. Si en algún "
        "momento la generación falla con un error 400 indicando que el modelo ya no "
        "existe, cambia la constante `GROQ_MODEL` en `src/ai_module.py`."
    )
