"""
ai_module.py
------------
Fase 3: Inteligencia Artificial con Groq.

Contiene funciones puras (sin dependencias de Streamlit) para:
  1. Resumir estadisticamente el subconjunto de datos que el usuario
     filtro en el dashboard (build_stat_summary).
  2. Construir el prompt que se envia al modelo (build_prompt).
  3. Llamar a la API de Groq (call_groq) y devolver el texto generado.

Separar esta logica de la UI permite probarla de forma aislada y
reutilizarla si en el futuro se cambia de proveedor de LLM.

NOTA SOBRE EL MODELO (importante para mantenimiento del proyecto):
Groq anuncio la baja de `llama-3.3-70b-versatile` con fecha de apagado
16-ago-2026 (ver https://console.groq.com/docs/deprecations),
recomendando migrar a `openai/gpt-oss-120b`. El challenge pide
explicitamente Llama 3.3 70B, asi que se deja como modelo por defecto,
pero se expone como constante `GROQ_MODEL` para poder cambiarlo con
una sola linea si el modelo deja de estar disponible.
"""

import json

import pandas as pd
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un consultor senior de datos presentando hallazgos a la junta directiva "
    "de TechLogistics S.A.S., un retailer tecnológico. Recibes un resumen estadístico "
    "de un subconjunto de datos ya filtrado por el usuario del dashboard (ventas, "
    "márgenes, logística y satisfacción del cliente). Tu tarea es escribir EXACTAMENTE "
    "TRES PÁRRAFOS de recomendación estratégica, en español, dirigidos a la junta "
    "directiva. Cada párrafo debe: (1) referirse a una dimensión distinta del negocio "
    "(rentabilidad, operación logística, experiencia del cliente), (2) basarse "
    "estrictamente en las cifras entregadas, sin inventar datos que no estén en el "
    "resumen, y (3) terminar con una recomendación accionable concreta. No uses "
    "encabezados, listas ni markdown; solo tres párrafos de texto corrido separados "
    "por un salto de línea."
)


class GroqAPIError(RuntimeError):
    """Error al invocar la API de Groq, con un mensaje ya listo para mostrar en la UI."""


def build_stat_summary(df: pd.DataFrame) -> dict:
    """
    Construye un resumen estadistico compacto del dataframe filtrado
    (la tabla 'ventas' ya unificada). Se diseño para ser JSON-serializable
    y suficientemente compacto para cualquier ventana de contexto.
    """
    n = len(df)
    if n == 0:
        return {
            "n_transacciones": 0,
            "nota": "No hay datos para los filtros seleccionados.",
        }

    try:
        ingreso_total = float(df["Ingreso_Total"].sum())
        margen_valido = df["Margen_Utilidad_USD"].dropna()
        margen_pct_valido = df["Margen_Utilidad_Pct"].dropna()

        n_fantasma = int(df["Es_SKU_Fantasma"].sum())
        ingreso_fantasma = float(df.loc[df["Es_SKU_Fantasma"], "Ingreso_Total"].sum())

        brecha = df["Brecha_Entrega_Dias"].dropna()

        top_categorias = (
            df.groupby("Categoria")["Ingreso_Total"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )
        top_ciudades = (
            df.groupby("Ciudad_Destino")["Ingreso_Total"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .round(2)
            .to_dict()
        )

        pct_feedback_no_confiable = None
        if "Feedback_Confiable" in df.columns:
            pct_feedback_no_confiable = round(
                float(df["Feedback_Confiable"].eq(False).mean() * 100), 2
            )

        fecha_min = df["Fecha_Venta"].min()
        fecha_max = df["Fecha_Venta"].max()

        resumen = {
            "n_transacciones": int(n),
            "ingreso_total_usd": round(ingreso_total, 2),
            "margen_total_usd": (
                round(float(margen_valido.sum()), 2) if len(margen_valido) else None
            ),
            "margen_mediano_pct": (
                round(float(margen_pct_valido.median()), 2) if len(margen_pct_valido) else None
            ),
            "pct_ventas_margen_negativo": (
                round(float((margen_valido < 0).mean() * 100), 2) if len(margen_valido) else None
            ),
            "pct_transacciones_sku_fantasma": round(n_fantasma / n * 100, 2),
            "ingreso_en_riesgo_sku_fantasma_usd": round(ingreso_fantasma, 2),
            "brecha_entrega_promedio_dias": (
                round(float(brecha.mean()), 2) if len(brecha) else None
            ),
            "pct_entregas_retrasadas": (
                round(float((brecha > 0).mean() * 100), 2) if len(brecha) else None
            ),
            "top_5_categorias_por_ingreso": top_categorias,
            "top_5_ciudades_por_ingreso": top_ciudades,
            "distribucion_segmento_nps": df["Segmento_NPS"].value_counts().to_dict(),
            "ratio_soporte_pct_promedio": round(
                float(df["Ticket_Soporte_Abierto"].mean() * 100), 2
            ),
            "pct_feedback_no_confiable": pct_feedback_no_confiable,
            "rango_fechas": {
                "desde": str(fecha_min.date()) if pd.notna(fecha_min) else None,
                "hasta": str(fecha_max.date()) if pd.notna(fecha_max) else None,
            },
        }
        return resumen

    except KeyError as exc:
        raise KeyError(
            f"build_stat_summary: falta la columna {exc} en el dataframe filtrado. "
            f"¿Se construyó con build_single_source_of_truth?"
        ) from exc


def build_prompt(resumen: dict) -> str:
    """Convierte el resumen estadístico en el mensaje de usuario para el LLM."""
    return (
        "Este es el resumen estadístico de los datos actualmente filtrados en el "
        "dashboard de TechLogistics S.A.S.:\n\n"
        f"{json.dumps(resumen, ensure_ascii=False, indent=2)}\n\n"
        "Escribe los tres párrafos de recomendación estratégica solicitados."
    )


def call_groq(api_key: str, resumen: dict, model: str = GROQ_MODEL, timeout: int = 30) -> str:
    """
    Llama a la API de Groq (compatible con OpenAI) y devuelve el texto generado.
    Lanza GroqAPIError con un mensaje entendible para mostrar en la UI si algo falla.
    """
    if not api_key or not api_key.strip():
        raise GroqAPIError("No se proporcionó una API Key de Groq.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(resumen)},
        ],
        "temperature": 0.4,
        "max_tokens": 900,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise GroqAPIError("Groq no respondió a tiempo (timeout). Intenta de nuevo.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise GroqAPIError(
            "No se pudo conectar con la API de Groq. Verifica tu conexión a internet."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise GroqAPIError(f"Error de red inesperado llamando a Groq: {exc}") from exc

    if resp.status_code == 401:
        raise GroqAPIError(
            "API Key inválida o expirada. Verifica tu Groq API Key en la barra lateral."
        )
    if resp.status_code == 429:
        raise GroqAPIError(
            "Se alcanzó el límite de solicitudes de Groq (rate limit). "
            "Espera un momento e intenta de nuevo."
        )
    if resp.status_code == 400:
        try:
            detalle = resp.json().get("error", {}).get("message", resp.text)
        except (json.JSONDecodeError, ValueError):
            detalle = resp.text
        raise GroqAPIError(
            f"Groq rechazó la solicitud (400): {detalle}. Si el modelo '{model}' ya no "
            f"está disponible, prueba con 'openai/gpt-oss-120b'."
        )
    if resp.status_code != 200:
        raise GroqAPIError(f"Groq devolvió un error {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise GroqAPIError("Groq devolvió una respuesta que no es JSON válido.") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise GroqAPIError(f"Respuesta inesperada de Groq: {data}") from exc
