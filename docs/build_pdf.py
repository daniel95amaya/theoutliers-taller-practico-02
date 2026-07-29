"""
build_pdf.py
------------
Genera docs/Documento_Hallazgos_TechLogistics.pdf a partir del analisis de
las 5 Preguntas de Alta Gerencia. Las graficas incrustadas son EXACTAMENTE
las mismas figuras que produce src/analysis.py y que la app de Streamlit
renderiza con st.pyplot() en la pestaña "5 Preguntas Estratégicas" -- no son
capturas de pantalla del navegador, pero sí el mismo objeto visual generado
por el mismo código que corre dentro del dashboard.

Uso:
    python docs/build_pdf.py

Requiere: reportlab (ya listado en requirements.txt del proyecto raiz solo
si se desea regenerar el PDF; no es una dependencia de app.py).
"""

import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUTPUT_PDF = os.path.join(HERE, "Documento_Hallazgos_TechLogistics.pdf")

NAVY = colors.HexColor("#1e3a5f")
BLUE = colors.HexColor("#2563eb")
RED = colors.HexColor("#dc2626")
GREY = colors.HexColor("#6b7280")
LIGHT_GREY = colors.HexColor("#f3f4f6")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "TituloPortada", parent=styles["Title"], fontSize=26, textColor=NAVY,
    spaceAfter=8, alignment=1,
))
styles.add(ParagraphStyle(
    "SubtituloPortada", parent=styles["Normal"], fontSize=14, textColor=GREY,
    alignment=1, spaceAfter=4,
))
styles.add(ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=17, textColor=NAVY, spaceBefore=18, spaceAfter=8,
))
styles.add(ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=13, textColor=BLUE, spaceBefore=10, spaceAfter=6,
))
styles.add(ParagraphStyle(
    "Cuerpo", parent=styles["Normal"], fontSize=10.3, leading=15, alignment=TA_JUSTIFY,
    spaceAfter=8,
))
styles.add(ParagraphStyle(
    "Diagnostico", parent=styles["Cuerpo"], backColor=LIGHT_GREY, borderPadding=8,
    leftIndent=4, rightIndent=4,
))
styles.add(ParagraphStyle(
    "Recomendacion", parent=styles["Cuerpo"], textColor=NAVY, fontName="Helvetica-Bold",
))


def _tabla(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def _imagen(nombre, ancho=6.6 * inch):
    path = os.path.join(ASSETS, nombre)
    alto = ancho * 585 / 1560  # relacion de aspecto real de las figuras (1560x585 px)
    return Image(path, width=ancho, height=alto)


def build_story():
    story = []

    # ---------------- PORTADA ----------------
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph("Documento de Hallazgos", styles["TituloPortada"]))
    story.append(Paragraph(
        "TechLogistics S.A.S. — Informe de Consultoría Senior", styles["SubtituloPortada"]
    ))
    story.append(Paragraph(
        "Curaduría de Datos, Integración y Recomendaciones Estratégicas",
        styles["SubtituloPortada"],
    ))
    story.append(Spacer(1, 0.6 * inch))
    story.append(Paragraph(
        "Preparado para la Junta Directiva de TechLogistics S.A.S.", styles["SubtituloPortada"]
    ))
    story.append(Paragraph(
        "Challenge 02 · Fundamentos en Ciencia de Datos (Maestría) · Universidad EAFIT",
        styles["SubtituloPortada"],
    ))
    story.append(Spacer(1, 1.6 * inch))
    story.append(Paragraph(
        "Las gráficas de este documento son las mismas visualizaciones generadas por el "
        "módulo <b>src/analysis.py</b> del dashboard de Streamlit, renderizadas con "
        "st.pyplot() en la pestaña \"5 Preguntas Estratégicas\" — no son montajes: son el "
        "producto directo del mismo código que corre en la aplicación.",
        styles["SubtituloPortada"],
    ))
    story.append(PageBreak())

    # ---------------- METODOLOGÍA ----------------
    story.append(Paragraph("Nota Metodológica", styles["H1"]))
    story.append(Paragraph(
        "Este informe responde, con evidencia visual y estadística, las 5 preguntas "
        "obligatorias de alta gerencia sobre la Sola Fuente de Verdad construida a partir "
        "de los tres sistemas de TechLogistics (Inventario, Logística y Feedback de "
        "Clientes). Las cifras se calcularon sobre las 10,000 transacciones del dataset "
        "completo; el dashboard permite recalcularlas sobre cualquier subconjunto filtrado.",
        styles["Cuerpo"],
    ))
    story.append(Paragraph(
        "En varias de las preguntas se probó formalmente (correlación de Pearson, ANOVA, "
        "chi-cuadrado) si el patrón que la pregunta da por sentado realmente existe en los "
        "datos. En más de un caso la respuesta honesta es que <b>no hay evidencia "
        "estadística suficiente</b> para la narrativa esperada — y reportar eso, en vez de "
        "forzar una historia conveniente, es precisamente lo que se espera de una "
        "consultoría senior.",
        styles["Cuerpo"],
    ))
    story.append(Spacer(1, 0.15 * inch))

    # ================= PREGUNTA 1 =================
    story.append(Paragraph("1. Fuga de Capital y Rentabilidad", styles["H1"]))
    story.append(Paragraph(
        "¿Los SKU con margen negativo representan una pérdida aceptable por volumen, o "
        "una falla crítica de precios en el canal Online?",
        styles["Cuerpo"],
    ))
    story.append(_tabla([
        ["Métrica", "Valor"],
        ["Ventas con margen negativo", "3,237 de 8,249 catalogadas (39.2%)"],
        ["Pérdida total acumulada", "-$11,670,392 USD"],
        ["Margen neto de la compañía", "$14,275,545 USD"],
        ["SKU con ≥3 ventas en pérdida (patrón sistemático)", "433 de 1,626 SKU afectados"],
        ["Correlación Precio_Venta_Final vs. Costo_Unitario_USD", "r = 0.013 (prácticamente nula)"],
    ], col_widths=[3.6 * inch, 3.0 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_imagen("q1_margen_canal.png"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Diagnóstico:</b> no es una falla del canal Online — de hecho es el canal con "
        "<i>menor</i> proporción de ventas en pérdida (37.3% vs. 41.2% en Físico). La "
        "correlación entre precio y costo es prácticamente cero: es una falla sistémica de "
        "gobernanza de precios, no un problema de un canal específico ni una pérdida "
        "aceptable por volumen — equivale al 82% del margen neto actual de la compañía.",
        styles["Diagnostico"],
    ))
    story.append(Paragraph(
        "Recomendación (Prioridad Alta): bloquear la publicación de un Precio_Venta_Final "
        "por debajo de Costo_Unitario_USD + margen mínimo, empezando por los 433 SKU con "
        "pérdida sistemática.",
        styles["Recomendacion"],
    ))
    story.append(PageBreak())

    # ================= PREGUNTA 2 =================
    story.append(Paragraph("2. Crisis Logística y Cuellos de Botella", styles["H1"]))
    story.append(Paragraph(
        "¿En qué ciudades y bodegas la correlación entre Tiempo de Entrega y NPS bajo es "
        "más fuerte? ¿Qué zona requiere un cambio inmediato de operador?",
        styles["Cuerpo"],
    ))
    story.append(_tabla([
        ["Nivel de análisis", "Correlación Tiempo Entrega ↔ NPS"],
        ["Global", "r = 0.007"],
        ["Mejor por ciudad (Bucaramanga)", "r = -0.020"],
        ["Mejor por bodega (Norte)", "r = -0.017"],
        ["Mejor combinación Ciudad×Bodega (Bucaramanga-Occidente, n=55)", "r = -0.121"],
    ], col_widths=[4.4 * inch, 2.2 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_imagen("q2_logistica_nps.png"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Diagnóstico:</b> la pregunta asume una correlación fuerte que, honestamente, no "
        "existe en los datos (ANOVA Estado_Envío→NPS: p=0.32; chi² Bodega×Estado_Envío: "
        "p=0.10). Lo que sí es real: la tasa de envíos problemáticos (~50%) está distribuida "
        "casi por igual en las 6 bodegas — no hay un \"operador culpable\" aislado, es una "
        "falla operativa de toda la red logística.",
        styles["Diagnostico"],
    ))
    story.append(Paragraph(
        "Recomendación (Prioridad Alta): auditoría operativa transversal a los 6 nodos "
        "logísticos en vez de reemplazar un operador puntual; mejorar la captura de NPS "
        "(63.8% de las ventas no tienen feedback) antes de intentar correlacionar "
        "satisfacción con logística.",
        styles["Recomendacion"],
    ))
    story.append(PageBreak())

    # ================= PREGUNTA 3 =================
    story.append(Paragraph("3. Análisis de la Venta Invisible", styles["H1"]))
    story.append(Paragraph(
        "¿Cuál es el impacto financiero de las ventas sin SKU en inventario? ¿Qué "
        "porcentaje del ingreso total está en riesgo?",
        styles["Cuerpo"],
    ))
    story.append(_tabla([
        ["Métrica", "Valor"],
        ["Transacciones con SKU fantasma", "1,751 de 10,000 (17.51%)"],
        ["SKU fantasma únicos", "480"],
        ["Ingreso generado por SKU fantasma", "$13,131,809 USD"],
        ["Ingreso total de la compañía", "$75,251,242 USD"],
        ["% del ingreso total en riesgo", "17.45%"],
        ["Margen calculable sobre ese ingreso", "$0 — no determinable"],
    ], col_widths=[3.6 * inch, 3.0 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_imagen("q3_sku_fantasma.png"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Diagnóstico:</b> casi 1 de cada 5 pesos de ingreso corresponde a productos que "
        "el sistema de inventario no reconoce. La evidencia (rango de SKU contiguo y "
        "separado del maestro, formato 100% válido, frecuencia de venta comparable a "
        "productos catalogados) indica que son <b>productos nuevos no catalogados</b> "
        "(falla de sincronización ERP↔Ventas), no errores de digitación. Sin costo de "
        "referencia, la junta no puede saber si ese 17.45% del ingreso es rentable o no.",
        styles["Diagnostico"],
    ))
    story.append(Paragraph(
        "Recomendación (Prioridad Alta / Crítica): bloquear en el sistema de ventas la "
        "facturación de SKU inexistentes en el maestro, y catalogar retroactivamente los "
        "480 SKU fantasma para calcular su margen real.",
        styles["Recomendacion"],
    ))
    story.append(PageBreak())

    # ================= PREGUNTA 4 =================
    story.append(Paragraph("4. Diagnóstico de Fidelidad", styles["H1"]))
    story.append(Paragraph(
        "¿Existen categorías de producto con alta disponibilidad (stock alto) pero "
        "sentimiento de cliente negativo? ¿Es mala calidad de producto o sobrecosto?",
        styles["Cuerpo"],
    ))
    story.append(_tabla([
        ["Categoría", "Stock prom.", "NPS prom.", "Rating", "Markup"],
        ["Smartphones", "1,063.1 (más alto)", "-4.22 (más bajo)", "2.98", "125.7%"],
        ["Sin Categorizar", "1,026.2", "-2.09", "3.06", "116.0%"],
        ["Monitores", "972.6", "0.05", "2.97", "137.1%"],
        ["Accesorios", "1,032.3", "0.83", "2.98", "127.0%"],
        ["Laptops", "992.8", "2.35", "2.98", "137.2%"],
        ["Tablets", "1,033.4", "3.96", "3.02", "150.2%"],
    ], col_widths=[1.5 * inch, 1.3 * inch, 1.3 * inch, 1.1 * inch, 1.1 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_imagen("q4_paradoja_fidelidad.png"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Diagnóstico:</b> la paradoja existe en el ranking simple (Smartphones: más "
        "stock, peor NPS), pero no resiste una prueba estadística rigurosa: la diferencia "
        "de NPS entre categorías no es significativa (ANOVA p=0.20), y el rating de "
        "producto es esencialmente idéntico en las 6 categorías (p=0.97). Esto descarta "
        "tanto \"mala calidad\" como \"sobrecosto\" (Smartphones ni siquiera es la categoría "
        "con mayor markup). El patrón probablemente sea ruido estadístico amplificado por "
        "la baja cobertura de feedback.",
        styles["Diagnostico"],
    ))
    story.append(Paragraph(
        "Recomendación (Prioridad Media): invertir en aumentar la tasa de respuesta de la "
        "encuesta de satisfacción antes de intervenir el pricing o la calidad de "
        "Smartphones con base en esta hipótesis.",
        styles["Recomendacion"],
    ))
    story.append(PageBreak())

    # ================= PREGUNTA 5 =================
    story.append(Paragraph("5. Storytelling de Riesgo Operativo", styles["H1"]))
    story.append(Paragraph(
        "Relación entre la antigüedad de la Última Revisión de stock y la tasa de Tickets "
        "de Soporte. ¿Qué bodegas están operando a ciegas?",
        styles["Cuerpo"],
    ))
    story.append(_tabla([
        ["Bodega", "Antigüedad prom. última revisión", "Tasa de tickets"],
        ["Sur", "507.9 días (1.39 años)", "19.8%"],
        ["Norte", "517.4 días (1.42 años)", "20.7%"],
        ["ZONA_FRANCA", "523.2 días (1.43 años)", "19.6%"],
        ["Occidente", "530.3 días (1.45 años)", "22.2%"],
        ["BOD-EXT-99", "533.6 días (1.46 años)", "20.0%"],
    ], col_widths=[1.8 * inch, 3.0 * inch, 1.8 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_imagen("q5_ceguera_inventario.png"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "<b>Diagnóstico:</b> no hay correlación estadística entre antigüedad de revisión y "
        "tickets de soporte (r=0.024). Pero el hallazgo real y contundente es el nivel "
        "absoluto: <b>las 6 bodegas llevan en promedio entre 1.39 y 1.46 años sin una "
        "revisión física de stock, sin excepción</b>. Toda la red opera a ciegas por "
        "igual — y esto conecta directamente con los 60 registros de Stock_Actual negativo "
        "detectados en la Fase 1 de auditoría de calidad.",
        styles["Diagnostico"],
    ))
    story.append(Paragraph(
        "Recomendación (Prioridad Alta): instaurar un ciclo obligatorio de recuento físico "
        "trimestral en las 6 bodegas por igual — no se justifica priorizar una sobre otra.",
        styles["Recomendacion"],
    ))
    story.append(PageBreak())

    # ================= RESUMEN EJECUTIVO / PLAN DE ACCIÓN =================
    story.append(Paragraph("Resumen Ejecutivo — Plan de Acción Priorizado", styles["H1"]))
    story.append(_tabla([
        ["#", "Hallazgo", "Impacto", "Prioridad"],
        ["1", "Precio desligado del costo (r≈0) en 433 SKU con pérdida sistemática",
         "-$11.67M en pérdidas evitables", "Alta"],
        ["2", "17.45% del ingreso ($13.1M) sin costo verificable por falta de "
              "control de inventario",
         "Margen real desconocido sobre 1 de cada 5 pesos de ingreso", "Alta"],
        ["3", "Toda la red logística (6/6 bodegas) con ~50% de envíos "
              "problemáticos, sin operador \"peor\" identificable",
         "Riesgo operativo sistémico, no localizado", "Alta"],
        ["4", "Ninguna bodega ha revisado físicamente su stock en menos de "
              "1.4 años en promedio",
         "Base de los negativos contables ya detectados en Fase 1", "Alta"],
        ["5", "Cobertura de feedback insuficiente (63.8% de ventas sin encuesta) "
              "impide validar hipótesis de fidelidad por categoría",
         "Decisiones de pricing/calidad con alto riesgo de basarse en ruido",
         "Media"],
    ], col_widths=[0.3 * inch, 2.9 * inch, 2.4 * inch, 0.9 * inch]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "<b>Principio transversal:</b> tres de las cinco preguntas del challenge (2, 4 y 5) "
        "parten de una hipótesis de correlación que los datos no confirman. Antes de "
        "intervenir logística, pricing o catálogo por categoría con base en esas "
        "hipótesis, TechLogistics debería primero cerrar las brechas de captura de datos "
        "(feedback, revisión de stock) que hoy hacen casi imposible distinguir una señal "
        "real de ruido estadístico.",
        styles["Cuerpo"],
    ))
    story.append(PageBreak())

    # ================= PLAN DE ACCIÓN TÁCTICO (3 recomendaciones) =================
    story.append(Paragraph("Plan de Acción Táctico", styles["H1"]))
    story.append(Paragraph(
        "Tres recomendaciones tácticas, numeradas y priorizadas por complejidad de "
        "implementación (no por impacto de negocio — las tres son de impacto alto; lo "
        "que las distingue es qué tan rápido y barato es ejecutarlas).",
        styles["Cuerpo"],
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph(
        "1. Bloquear la venta de SKU sin catálogo oficial", styles["H2"]
    ))
    story.append(Paragraph(
        "<b>Complejidad: Baja.</b> Agregar una validación en el sistema de ventas que "
        "impida registrar una transacción con un SKU_ID inexistente en el maestro de "
        "inventario. Es un cambio de una sola regla de negocio, no requiere rediseñar "
        "ningún sistema. Impacto: detiene inmediatamente el crecimiento del 17.45% de "
        "ingreso hoy \"invisible\" para el control de inventario (Pregunta 3).",
        styles["Cuerpo"],
    ))
    story.append(Paragraph(
        "2. Regla de piso de precio ligada al costo real", styles["H2"]
    ))
    story.append(Paragraph(
        "<b>Complejidad: Media.</b> Requiere: (a) exponer Costo_Unitario_USD al motor "
        "de pricing en tiempo real, (b) definir el margen mínimo aceptable por "
        "categoría, y (c) un proceso de excepción para descuentos estratégicos "
        "autorizados. Es un cambio de integración entre sistemas, no solo de reglas. "
        "Impacto: ataca directamente los -$11.67M en pérdidas evitables (Pregunta 1).",
        styles["Cuerpo"],
    ))
    story.append(Paragraph(
        "3. Ciclo trimestral de recuento físico en las 6 bodegas", styles["H2"]
    ))
    story.append(Paragraph(
        "<b>Complejidad: Alta.</b> Requiere presupuesto operativo recurrente, "
        "coordinación logística en 6 ubicaciones simultáneamente, y un proceso de "
        "conciliación entre el conteo físico y el sistema. Es el cambio más costoso de "
        "implementar, pero corrige la causa raíz de los stocks negativos (Fase 1) y de "
        "la falta de visibilidad operativa (Pregunta 5) — sin esto, cualquier otra "
        "corrección de datos seguirá degradándose con el tiempo.",
        styles["Cuerpo"],
    ))

    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=letter,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        title="Documento de Hallazgos - TechLogistics S.A.S.",
        author="Consultoría Senior - Challenge 02 EAFIT",
    )
    doc.build(build_story())
    print(f"PDF generado en: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
