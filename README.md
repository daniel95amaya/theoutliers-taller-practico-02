# TechLogistics S.A.S. — Sistema de Soporte a la Decisión (DSS)

Challenge 02 · Fundamentos en Ciencia de Datos (Maestría) · EAFIT
Consultoría Senior: curaduría de datos, integración y recomendaciones estratégicas con IA.

**🔗 App en vivo:** `<PENDIENTE — pega aquí la URL que te da Streamlit Community Cloud al
desplegar, ej. https://tu-usuario-techlogistics.streamlit.app>`
> Ver la sección "Desplegar en Streamlit Community Cloud" más abajo. Después de desplegar,
> reemplaza esta línea con la URL real antes de entregar el repositorio.

## Descripción del problema

TechLogistics S.A.S. tiene tres sistemas que no se hablan entre sí (ERP de Inventarios,
Logística y Feedback de clientes). Este proyecto limpia, audita, integra y analiza esos
tres datasets para construir un dashboard interactivo que responde las preguntas de alta
gerencia sobre rentabilidad, logística y satisfacción del cliente, y genera recomendaciones
estratégicas en tiempo real usando el modelo Llama 3.3 70B a través de la API de Groq.

## Estructura del repositorio

```
.
├── app.py                  # App principal de Streamlit (punto de entrada)
├── requirements.txt        # Dependencias para Streamlit Community Cloud
├── data/                   # Datasets crudos (CSV) provistos por el challenge
│   ├── inventario_central_v2.csv
│   ├── transacciones_logistica_v2.csv
│   └── feedback_clientes_v2.csv
├── docs/
│   ├── Documento_Hallazgos_TechLogistics.pdf   # Informe de consultoría (PDF, para la junta)
│   ├── build_pdf.py                            # Script que genera el PDF (reportlab)
│   └── assets/                                 # Gráficas (PNG) incrustadas en el PDF
└── src/                    # Lógica de negocio, separada de la UI
    ├── quality.py          # Fase 1: nulidad, duplicados, outliers (IQR), health score
    ├── cleaning.py         # Fase 1: limpieza e imputación (decisiones documentadas)
    ├── integration.py      # Fase 2: merge estratégico + variables derivadas
    ├── analysis.py         # Las 5 Preguntas de Alta Gerencia (recalculadas sobre cualquier filtro)
    └── ai_module.py         # Fase 3: resumen estadístico + integración con Groq
```

## Documento de Hallazgos (PDF)

`docs/Documento_Hallazgos_TechLogistics.pdf` es el informe de consultoría dirigido a la
junta directiva: narrativa de negocio, tablas de evidencia y las 5 gráficas de la pestaña
"5 Preguntas Estratégicas" del dashboard (las mismas figuras que genera `src/analysis.py`
y que la app renderiza con `st.pyplot()` — no son montajes, es el mismo código).

Para regenerarlo (por ejemplo, tras actualizar el análisis):

```bash
pip install reportlab
python docs/build_pdf.py
```

Esto sobrescribe `docs/Documento_Hallazgos_TechLogistics.pdf` usando las imágenes en
`docs/assets/`. `reportlab` solo es necesario para regenerar el PDF — no es una
dependencia de `app.py` y por eso no está en `requirements.txt` del dashboard.

## Pestañas del dashboard

1. **🔍 Auditoría de Calidad** — Health Score antes/después, nulidad, duplicados, outliers y el dilema del SKU fantasma (Fase 1).
2. **📦 Operaciones** — margen de utilidad, ingreso por ciudad, brecha de entrega, ratio de soporte (Fase 2).
3. **👥 Cliente** — segmentación NPS, rating por categoría, alerta de feedback no confiable.
4. **📋 5 Preguntas Estratégicas** — responde en vivo, con gráficas, las 5 preguntas obligatorias del challenge (fuga de capital, crisis logística, venta invisible, paradoja de fidelidad, ceguera de inventario), recalculadas sobre los filtros activos en la barra lateral.
5. **🤖 Insights de IA** — recomendaciones generadas por Llama 3.3 70B (Groq) sobre el resumen estadístico filtrado.

Ver `docs/Documento_Hallazgos_TechLogistics.pdf` para el análisis completo de la pestaña 4, con las pruebas estadísticas formales (correlación, ANOVA, chi-cuadrado) que sustentan cada conclusión sobre el dataset completo.

## Ejecutar localmente

```bash
git clone <url-del-repo>
cd <nombre-del-repo>
python -m venv .venv
source .venv/bin/activate       # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La app abrirá en `http://localhost:8501`.

## Desplegar en Streamlit Community Cloud (desde GitHub)

1. Sube este repositorio a GitHub (público o privado con acceso de Streamlit).
2. Entra a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
3. Click en **"New app"** → selecciona el repositorio, la rama (`main`) y como
   **Main file path** escribe: `app.py`.
4. Click en **"Deploy"**. La primera vez tarda unos minutos en instalar las dependencias
   de `requirements.txt`.
5. Los datasets ya están incluidos en `data/`, así que la app funciona sin configuración
   adicional — **no necesitas subir los CSV por separado ni configurar `st.secrets`**.

## Uso del módulo de IA (Groq / Llama 3.3 70B)

La app **no** trae una API Key preconfigurada. Cada usuario debe ingresar la suya:

1. Consigue una API Key gratuita en [console.groq.com/keys](https://console.groq.com/keys).
2. En el dashboard, pégala en el campo **"🤖 Groq API Key"** de la barra lateral
   (queda oculta con formato contraseña).
3. Ve a la pestaña **"🤖 Insights de IA"**, revisa el resumen estadístico de los datos que
   tienes filtrados, y presiona **"✨ Generar Recomendaciones Estratégicas"**.
4. La API Key **no se guarda** en ningún archivo, base de datos ni log de la aplicación:
   vive solo en la sesión del navegador y se envía directamente a la API de Groq.

> ⚠️ **Nota de mantenimiento:** Groq anunció el retiro (deprecación) del modelo
> `llama-3.3-70b-versatile` con fecha de apagado el **16 de agosto de 2026**, recomendando
> migrar a `openai/gpt-oss-120b`. El challenge exige explícitamente Llama 3.3 70B, así que
> se dejó como modelo por defecto. Si en el futuro las solicitudes empiezan a fallar con un
> error 400 de "modelo no encontrado", solo hay que cambiar la constante `GROQ_MODEL` en
> `src/ai_module.py` por el modelo de reemplazo.

## Principios de diseño

- **Transparencia radical**: ninguna limpieza se hace sin dejar rastro. La pestaña
  "Auditoría de Calidad" muestra el Health Score antes/después, la nulidad por columna y
  la bitácora completa de cada decisión de imputación.
- **No se ocultan los problemas incómodos**: las ventas con SKU fantasma (17.5% del
  ingreso) no se eliminan del análisis — se marcan y se reportan explícitamente, igual que
  el feedback con `Transaccion_ID` colisionado.
- **Código modular**: la lógica de limpieza, integración e IA vive en `src/`, separada de
  la interfaz en `app.py`, para que se pueda probar y reutilizar de forma independiente.

## Buenas prácticas de código

- **PEP8**: todas las líneas ≤99 caracteres (límite documentado y verificado con análisis
  estático; PEP8 permite este ajuste de equipo sobre el límite estricto de 79 cuando el
  proyecto lo declara). Nombres de función/variable en `snake_case`, imports al inicio de
  cada archivo, dos líneas en blanco entre funciones de nivel superior.
- **Manejo de excepciones**: cada módulo de `src/` define su propia excepción de negocio
  (`DataCleaningError`, `IntegrationError`, `AnalysisError`, `GroqAPIError`,
  `QualityReportError`) y las relanza con `raise ... from exc` para conservar la traza
  original. `app.py` nunca deja que un error de datos tumbe la app con un traceback crudo:
  cada sección del dashboard atrapa su excepción específica y muestra un mensaje entendible
  con `st.error(...)`.
- **Separación de responsabilidades**: `app.py` solo contiene lógica de interfaz
  (Streamlit); toda la lógica de negocio (limpieza, integración, análisis, llamadas a la
  API de Groq) vive en `src/`, es importable y testeable de forma independiente.

## Autor / Curso

Fundamentos en Ciencia de Datos (Maestría) — Universidad EAFIT, Periodo 2026-1
Docente: Jorge Iván Padilla-Buriticá
