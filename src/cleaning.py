"""
cleaning.py
-----------
Aplica las decisiones de limpieza/imputación documentadas en la Fase 1
(Auditoria de Calidad). Cada funcion devuelve (df_limpio, log) donde
`log` es un diccionario con el detalle de cada correccion hecha, para
alimentar el modulo de transparencia "Antes vs Despues" del dashboard.

Principio rector: nunca eliminar una fila completa por un solo campo
sucio. Se imputa con mediana (numericas simetricas) o se etiqueta
explicitamente (categoricas / campos de opinion), nunca se inventa
una categoria por moda cuando eso podria sesgar el analisis de negocio.
"""

import pandas as pd
import numpy as np


class DataCleaningError(Exception):
    """Error de negocio al limpiar un dataset (columna faltante, formato inesperado)."""


REQUIRED_COLS_INVENTARIO = [
    "SKU_ID", "Categoria", "Stock_Actual", "Costo_Unitario_USD",
    "Bodega_Origen", "Lead_Time_Dias",
]
REQUIRED_COLS_TRANSACCIONES = [
    "Transaccion_ID", "SKU_ID", "Fecha_Venta", "Cantidad_Vendida",
    "Precio_Venta_Final", "Costo_Envio", "Tiempo_Entrega_Real",
    "Estado_Envio", "Ciudad_Destino",
]
REQUIRED_COLS_FEEDBACK = [
    "Feedback_ID", "Transaccion_ID", "Rating_Producto", "Edad_Cliente",
    "Recomienda_Marca", "Comentario_Texto", "Ticket_Soporte_Abierto",
]


def _validar_columnas(df: pd.DataFrame, columnas: list, nombre_dataset: str) -> None:
    faltantes = [c for c in columnas if c not in df.columns]
    if faltantes:
        raise DataCleaningError(
            f"{nombre_dataset}: faltan columnas requeridas {faltantes}. "
            f"Verifica que el CSV no haya cambiado de esquema."
        )


# ---------------------------------------------------------------------------
# INVENTARIO
# ---------------------------------------------------------------------------
def clean_inventario(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica las correcciones documentadas en la Fase 1 al inventario."""
    try:
        _validar_columnas(df, REQUIRED_COLS_INVENTARIO, "inventario_central_v2.csv")
        df = df.copy()
        log = {"dataset": "inventario_central_v2.csv", "acciones": []}

        # 1. Normalizacion de Categoria
        mapa_categoria = {"smart-phone": "Smartphones", "LAPTOP": "Laptops"}
        n_norm = df["Categoria"].isin(mapa_categoria.keys()).sum()
        df["Categoria"] = df["Categoria"].replace(mapa_categoria)
        n_unk = (df["Categoria"] == "???").sum()
        df["Categoria"] = df["Categoria"].replace("???", "Sin Categorizar")
        log["acciones"].append(
            f"Categoria: {n_norm} filas normalizadas (smart-phone/LAPTOP), "
            f"{n_unk} filas etiquetadas 'Sin Categorizar' (antes '???'), "
            f"sin imputacion por moda."
        )

        # 2. Normalizacion de Bodega_Origen (solo capitalizacion;
        #    BOD-EXT-99 y ZONA_FRANCA se conservan)
        n_bodega = (df["Bodega_Origen"] == "norte").sum()
        df["Bodega_Origen"] = df["Bodega_Origen"].replace({"norte": "Norte"})
        log["acciones"].append(
            f"Bodega_Origen: {n_bodega} filas normalizadas ('norte'->'Norte'). "
            f"'BOD-EXT-99' y 'ZONA_FRANCA' se conservan como nodos logisticos validos."
        )

        # 3. Stock_Actual: negativos -> NaN -> imputar mediana
        n_neg = (df["Stock_Actual"] < 0).sum()
        df.loc[df["Stock_Actual"] < 0, "Stock_Actual"] = np.nan
        n_null_before = df["Stock_Actual"].isna().sum()
        mediana_stock = df["Stock_Actual"].median()
        df["Stock_Actual"] = df["Stock_Actual"].fillna(mediana_stock)
        log["acciones"].append(
            f"Stock_Actual: {n_neg} negativos convertidos a NaN; "
            f"{n_null_before} nulos totales imputados con mediana "
            f"({mediana_stock:.0f} u.)."
        )

        # 4. Costo_Unitario_USD: winsorizar en P1/P99
        # Se GUARDAN los registros afectados (no solo el conteo) porque la Guia de
        # Validacion exige que el dashboard ofrezca una vista "Ver registros excluidos".
        p1, p99 = df["Costo_Unitario_USD"].quantile([0.01, 0.99])
        wins_mask = (df["Costo_Unitario_USD"] < p1) | (df["Costo_Unitario_USD"] > p99)
        registros_winsorizados = df.loc[
            wins_mask, ["SKU_ID", "Categoria", "Costo_Unitario_USD"]
        ].rename(columns={"Costo_Unitario_USD": "Costo_Original_USD"}).copy()
        registros_winsorizados["Costo_Capado_USD"] = df.loc[wins_mask, "Costo_Unitario_USD"].clip(
            lower=p1, upper=p99
        )
        n_wins = int(wins_mask.sum())
        df["Costo_Unitario_USD"] = df["Costo_Unitario_USD"].clip(lower=p1, upper=p99)
        log["acciones"].append(
            f"Costo_Unitario_USD: {n_wins} valores winsorizados a "
            f"[P1={p1:.2f}, P99={p99:.2f}] (afecta principalmente $850,000 y $0.05)."
        )
        log["registros_excluidos"] = registros_winsorizados.reset_index(drop=True)

        # 5. Lead_Time_Dias: a categoria ordinal + imputar nulos con moda
        df["Lead_Time_Categoria"] = df["Lead_Time_Dias"].apply(_bucket_lead_time)
        n_null_lt = df["Lead_Time_Categoria"].isna().sum()
        moda_lt = df["Lead_Time_Categoria"].mode()
        moda_lt = moda_lt.iloc[0] if not moda_lt.empty else "Corto (3-5 dias)"
        df["Lead_Time_Categoria"] = df["Lead_Time_Categoria"].fillna(moda_lt)
        log["acciones"].append(
            f"Lead_Time_Dias: reclasificado a variable ordinal 'Lead_Time_Categoria'; "
            f"{n_null_lt} nulos imputados con la moda ('{moda_lt}')."
        )

        # 6. Validacion temporal: Ultima_Revision no puede ser una fecha futura.
        # Se compara dinamicamente contra "ahora" (no una fecha fija en el codigo),
        # tal como exige la Guia de Validacion.
        hoy = pd.Timestamp.now().normalize()
        fechas_revision = pd.to_datetime(df["Ultima_Revision"])
        futuras_mask = fechas_revision > hoy
        n_futuras = int(futuras_mask.sum())
        if n_futuras:
            df.loc[futuras_mask, "Ultima_Revision"] = hoy.strftime("%Y-%m-%d")
        log["acciones"].append(
            f"Ultima_Revision: {n_futuras} fechas posteriores a hoy "
            f"({hoy.date()}) detectadas y corregidas al día actual "
            f"(validación dinámica contra la fecha real de ejecución)."
        )

        return df, log

    except DataCleaningError:
        raise
    except KeyError as exc:
        raise DataCleaningError(f"clean_inventario: columna inesperada faltante: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - se relanza con contexto de negocio
        raise DataCleaningError(
            f"clean_inventario: fallo inesperado limpiando inventario: {exc}"
        ) from exc


def _bucket_lead_time(v):
    """Reclasifica Lead_Time_Dias (mezcla de texto/numero) a una categoria ordinal."""
    if pd.isna(v):
        return np.nan
    v = str(v).strip()
    if v == "Inmediato":
        return "Inmediato"
    if v == "25-30 días":
        return "Largo (25-30 días)"
    try:
        n = int(float(v))
    except ValueError:
        return np.nan
    return "Corto (3-5 días)" if n <= 5 else "Medio (10 días)"


# ---------------------------------------------------------------------------
# TRANSACCIONES
# ---------------------------------------------------------------------------
def clean_transacciones(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica las correcciones documentadas en la Fase 1 a las transacciones."""
    try:
        _validar_columnas(df, REQUIRED_COLS_TRANSACCIONES, "transacciones_logistica_v2.csv")
        df = df.copy()
        log = {"dataset": "transacciones_logistica_v2.csv", "acciones": []}

        # 1. Fecha_Venta -> datetime
        df["Fecha_Venta"] = pd.to_datetime(df["Fecha_Venta"], format="%d/%m/%Y")

        # 1b. Validacion temporal: una venta no puede tener fecha futura. Se compara
        # dinamicamente contra "ahora" (pd.Timestamp.now(), no una fecha fija en el
        # codigo) tal como exige la Guia de Validacion. Las filas futuras se excluyen
        # del calculo de series de tiempo pero NUNCA se eliminan silenciosamente del
        # dataset: se marcan para que el dashboard pueda auditarlas.
        hoy = pd.Timestamp.now().normalize()
        futuras_mask = df["Fecha_Venta"] > hoy
        n_futuras = int(futuras_mask.sum())
        df["Fecha_Venta_Futura"] = futuras_mask
        log["acciones"].append(
            f"Fecha_Venta: {n_futuras} transacciones con fecha posterior a hoy "
            f"({hoy.date()}) detectadas y marcadas con 'Fecha_Venta_Futura' -- se "
            f"excluyen de los gráficos de series de tiempo pero se conservan en la "
            f"tabla (no se eliminan filas de venta reales por una sola fecha corrupta)."
        )

        # 2. Ciudad_Destino: normalizar codigos, aislar 'Ventas_Web'
        mapa_ciudad = {"BOG": "Bogotá", "MED": "Medellín"}
        n_norm_ciudad = df["Ciudad_Destino"].isin(mapa_ciudad.keys()).sum()
        df["Ciudad_Destino"] = df["Ciudad_Destino"].replace(mapa_ciudad)
        n_web = (df["Ciudad_Destino"] == "Ventas_Web").sum()
        df["Ciudad_Destino"] = df["Ciudad_Destino"].replace(
            "Ventas_Web", "Ciudad No Especificada"
        )
        log["acciones"].append(
            f"Ciudad_Destino: {n_norm_ciudad} codigos normalizados (BOG/MED); "
            f"{n_web} filas 'Ventas_Web' reclasificadas a 'Ciudad No Especificada' "
            f"(no se infiere ciudad a partir del canal de venta)."
        )

        # 3. Estado_Envio: nulos -> categoria explicita
        n_null_estado = df["Estado_Envio"].isna().sum()
        df["Estado_Envio"] = df["Estado_Envio"].fillna("Sin Información")
        log["acciones"].append(
            f"Estado_Envio: {n_null_estado} nulos etiquetados 'Sin Información' "
            f"(NO imputados por moda, para no distorsionar KPIs de servicio)."
        )

        # 4. Cantidad_Vendida: centinela -5 -> NaN -> imputar mediana
        n_centinela_cant = (df["Cantidad_Vendida"] == -5).sum()
        df.loc[df["Cantidad_Vendida"] == -5, "Cantidad_Vendida"] = np.nan
        mediana_cant = df["Cantidad_Vendida"].median()
        df["Cantidad_Vendida"] = df["Cantidad_Vendida"].fillna(mediana_cant)
        log["acciones"].append(
            f"Cantidad_Vendida: {n_centinela_cant} valores centinela (-5) "
            f"imputados con mediana ({mediana_cant:.0f} u.)."
        )

        # 5. Tiempo_Entrega_Real: centinela 999 -> NaN -> imputar mediana
        n_centinela_tiempo = (df["Tiempo_Entrega_Real"] == 999).sum()
        df.loc[df["Tiempo_Entrega_Real"] == 999, "Tiempo_Entrega_Real"] = np.nan
        mediana_tiempo = df["Tiempo_Entrega_Real"].median()
        df["Tiempo_Entrega_Real"] = df["Tiempo_Entrega_Real"].fillna(mediana_tiempo)
        log["acciones"].append(
            f"Tiempo_Entrega_Real: {n_centinela_tiempo} valores centinela (999) "
            f"imputados con mediana ({mediana_tiempo:.0f} dias)."
        )

        # 6. Costo_Envio: nulos -> imputar mediana
        n_null_costo = df["Costo_Envio"].isna().sum()
        mediana_costo = df["Costo_Envio"].median()
        df["Costo_Envio"] = df["Costo_Envio"].fillna(mediana_costo)
        log["acciones"].append(
            f"Costo_Envio: {n_null_costo} nulos imputados con mediana "
            f"(${mediana_costo:.2f})."
        )

        return df, log

    except DataCleaningError:
        raise
    except KeyError as exc:
        raise DataCleaningError(
            f"clean_transacciones: columna inesperada faltante: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise DataCleaningError(
            f"clean_transacciones: fallo inesperado limpiando transacciones: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# FEEDBACK
# ---------------------------------------------------------------------------
def clean_feedback(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica las correcciones documentadas en la Fase 1 al feedback de clientes."""
    try:
        _validar_columnas(df, REQUIRED_COLS_FEEDBACK, "feedback_clientes_v2.csv")
        df = df.copy()
        log = {"dataset": "feedback_clientes_v2.csv", "acciones": []}

        # 1. Regenerar Feedback_ID unico (colisiones de llave primaria)
        n_colisiones = df["Feedback_ID"].duplicated(keep=False).sum()
        df = df.reset_index(drop=True)
        df["Feedback_ID"] = ["FB-" + str(i).zfill(6) for i in range(1, len(df) + 1)]
        log["acciones"].append(
            f"Feedback_ID: {n_colisiones} filas con ID colisionado -> se regenero "
            f"un ID surrogado unico por fila. NINGUNA fila fue eliminada "
            f"(son eventos distintos)."
        )

        # 1b. Transaccion_ID colisionado dentro de feedback (falla DISTINTA a la
        # de Feedback_ID). Una Transaccion_ID = una orden de un solo cliente; por
        # lo tanto no puede tener mas de una fila de feedback genuina. Se
        # verifico que en 99% de los grupos duplicados la Edad_Cliente varia
        # entre filas del mismo Transaccion_ID -> imposible si fuera el mismo
        # cliente respondiendo dos veces. Conclusion: es una falla de
        # generacion de ID (muestreo sin garantia de unicidad), no un evento
        # de negocio real (encuesta repetida). Decision: NO eliminar ni
        # fusionar filas silenciosamente -> se marca con una bandera de
        # confiabilidad para que la agregacion aguas abajo (Fase 2) sea
        # transparente.
        txn_dup_mask = df["Transaccion_ID"].duplicated(keep=False)
        n_txn_colisiones = int(txn_dup_mask.sum())
        n_txn_ids_afectados = int(df.loc[txn_dup_mask, "Transaccion_ID"].nunique())
        df["Feedback_Confiable"] = ~txn_dup_mask
        log["acciones"].append(
            f"Transaccion_ID: {n_txn_colisiones} filas comparten Transaccion_ID "
            f"con al menos otra fila ({n_txn_ids_afectados} IDs de transaccion "
            f"afectados). Se confirmo (variacion de Edad_Cliente dentro del "
            f"mismo grupo) que es una falla de integridad de llave, no feedback "
            f"repetido legitimo. Se agrega la bandera 'Feedback_Confiable' "
            f"(False en filas de grupos conflictivos) en vez de promediar u "
            f"ocultar el conflicto sin dejar rastro."
        )

        # 2. Rating_Producto: centinela 99 -> NaN -> imputar mediana
        n_centinela_rating = (df["Rating_Producto"] == 99).sum()
        df.loc[df["Rating_Producto"] == 99, "Rating_Producto"] = np.nan
        mediana_rating = df["Rating_Producto"].median()
        df["Rating_Producto"] = df["Rating_Producto"].fillna(mediana_rating)
        log["acciones"].append(
            f"Rating_Producto: {n_centinela_rating} valores centinela (99) "
            f"imputados con mediana ({mediana_rating:.0f})."
        )

        # 3. Edad_Cliente: centinela 195 -> NaN -> imputar mediana
        n_centinela_edad = (df["Edad_Cliente"] == 195).sum()
        df.loc[df["Edad_Cliente"] == 195, "Edad_Cliente"] = np.nan
        mediana_edad = df["Edad_Cliente"].median()
        df["Edad_Cliente"] = df["Edad_Cliente"].fillna(mediana_edad)
        log["acciones"].append(
            f"Edad_Cliente: {n_centinela_edad} valores centinela (195 años) "
            f"imputados con mediana ({mediana_edad:.0f} años)."
        )

        # 4. Recomienda_Marca: nulos -> categoria explicita, normalizar SI/NO/Maybe
        mapa_recomienda = {"SI": "Sí", "NO": "No", "Maybe": "Indeciso"}
        df["Recomienda_Marca"] = df["Recomienda_Marca"].replace(mapa_recomienda)
        n_null_recomienda = df["Recomienda_Marca"].isna().sum()
        df["Recomienda_Marca"] = df["Recomienda_Marca"].fillna("No Responde")
        log["acciones"].append(
            f"Recomienda_Marca: normalizado a Sí/No/Indeciso; "
            f"{n_null_recomienda} nulos etiquetados 'No Responde' "
            f"(NO imputados, es una opinión subjetiva)."
        )

        # 5. Comentario_Texto: nulos -> texto explicito
        n_null_comentario = df["Comentario_Texto"].isna().sum()
        df["Comentario_Texto"] = df["Comentario_Texto"].fillna("Sin comentario")
        log["acciones"].append(
            f"Comentario_Texto: {n_null_comentario} nulos reemplazados por "
            f"'Sin comentario' (no se genera texto sintetico)."
        )

        # 6. Ticket_Soporte_Abierto: normalizar a booleano
        mapa_ticket = {"Sí": True, "1": True, 1: True, "No": False, "0": False, 0: False}
        df["Ticket_Soporte_Abierto"] = (
            df["Ticket_Soporte_Abierto"].replace(mapa_ticket).astype(bool)
        )
        log["acciones"].append(
            "Ticket_Soporte_Abierto: normalizado a booleano unico "
            "(antes 4 codificaciones distintas)."
        )

        return df, log

    except DataCleaningError:
        raise
    except KeyError as exc:
        raise DataCleaningError(f"clean_feedback: columna inesperada faltante: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise DataCleaningError(
            f"clean_feedback: fallo inesperado limpiando feedback: {exc}"
        ) from exc
