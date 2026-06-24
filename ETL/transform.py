"""
Transformaciones de limpieza y normalización para cada fuente de datos.

Convenciones
------------
- fuente: "Fuente A" | "Fuente B" | "Fuente C" (identificadores anonimizados)
- columna_id: código de equipo tal como viene del LIMS (ej. "C064-31")
- tipo_columna: prefijo antes del primer guion (ej. "C064")
- distancia_umbral: (actual - umbral) / |umbral|, normalizada por dirección del comparador.
  Positivo = margen de seguridad; cero = en el límite; negativo = fallo.
"""
import pandas as pd

# Parámetros operacionales que no son parte del SST evaluable.
PARAMETROS_OPERACIONALES = {
    "ALMACENAMIENTO DE COLUMNA",
    "ALMACENAMIENTO DE CAPILAR",
    "LAVADO DE COLUMNA",
    "LAVADO DE CAPILAR",
    "PRECORRIDA",
}

# Motivos de baja que indican un fallo real del SST (no obsolescencia ni corrección).
MOTIVOS_FALLO_SST = {
    "NO CUMPLE ADECUACION",
    "NO CUMPLE RESOLUCION",
}

# Comparadores en los que "bien" significa actual > umbral (ej. N >= 8000).
COMPARADORES_MAYOR = {"greater", "greater_equal"}
# Comparadores en los que "bien" significa actual < umbral (ej. Tailing <= 2).
COMPARADORES_MENOR = {"less", "less_equal"}

# Unificación de nombres de parámetros SST entre fuentes y variantes de nomenclatura.
UNIFICACION_PARAMETROS = {
    # Platos teóricos
    "Platos teoricos":            "Platos teóricos",
    "Platos teóricos (mínimo)":   "Platos teóricos",
    "PLATOS TEORICOS ACTIVO 1":   "Platos teóricos",
    # Factor tailing / coleo / simetría — mismo concepto físico (asimetría de pico)
    "Factor de coleo":            "Factor tailing",
    "Factor de simetria":         "Factor tailing",
    "Factor de Tailing":          "Factor tailing",
    "Factor de Tailing (máximo)": "Factor tailing",
    # Precisión / RSD
    "PRECISION ACTIVO 1":         "Precisión",
    "PRECISION ACTIVO 2":         "Precisión",
    "PRECISION ACTIVO 3":         "Precisión",
    "RSD%":                       "Precisión",
    # Resolución
    "Resolución (mínima)":        "Resolución",
}

# Estado de columna: mapeo de códigos LIMS a etiquetas en español.
ESTADO_MAP = {
    "inStock":              "En stock",
    "inUse":                "En uso",
    "disposed":             "Baja",
    "disposal_in_process":  "En proceso de baja",
}

# Fragmentos de texto en las notas de baja de Fuente C que indican fallo SST real.
_KEYWORDS_FALLO_SST_NOTAS = [
    "no cumple",
    "no resuelve",
    "fuera del limite",
    "fuera del límite",
    "baja resoluci",
    "perfil cromatografico",
    "perfil cromatográfico",
    "sst",
    "factor de simetr",
    "factor de coleo",
    "platos",
]


def _detectar_fallo_sst_por_notas(notas: pd.Series) -> pd.Series:
    """Detecta fallos SST buscando keywords en el campo de notas de baja."""
    notas_lower = notas.fillna("").str.lower()
    return notas_lower.apply(
        lambda txt: any(kw in txt for kw in _KEYWORDS_FALLO_SST_NOTAS)
    )


# -- Fuente A ------------------------------------------------------------------

def transformar_columnas_hplc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el catálogo de columnas de Fuente A.

    Salida adicional vs. raw: tipo_columna, fallo_sst, censurada.
    Fechas parseadas a datetime.
    """
    df = df.copy()

    renombrar = {
        "Identificador":    "columna_id",
        "Nombre":           "nombre",
        "Marca":            "marca",
        "Nro. de serie":    "nro_serie",
        "Catálogo":         "catalogo",
        "Recepción":        "fecha_recepcion",
        "Costo ($)":        "costo",
        "Puesta en uso":    "fecha_puesta_en_uso",
        "Baja":             "fecha_baja",
        "Uso asignado":     "uso_asignado",
        "Ubicación":        "ubicacion",
        "Lavado":           "lavado",
        "Almacenamiento":   "almacenamiento",
        "Estado":           "estado",
        "Notas":            "notas",
        "Motivo baja":      "motivo_baja",
        "Último uso":       "fecha_ultimo_uso",
        "Total inyecciones": "total_inyecciones",
    }
    df = df.rename(columns={c: renombrar[c] for c in df.columns if c in renombrar})

    for col in ["fecha_recepcion", "fecha_puesta_en_uso", "fecha_baja", "fecha_ultimo_uso"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    df["tipo_columna"] = df["columna_id"].str.split("-").str[0]
    df["fallo_sst"]    = df["motivo_baja"].isin(MOTIVOS_FALLO_SST)
    df["censurada"]    = ~df["fallo_sst"]

    return df


def transformar_resultados_criterios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza la tabla de criterios SST.

    Aplica:
      - Filtro de parámetros operacionales
      - Exclusión de comparadores no evaluables (no_compare)
      - Exclusión de filas sin valor medido
      - Cálculo de distancia_umbral normalizada
      - Unificación de nombres de parámetros entre fuentes
    """
    df = df.copy()

    df = df.rename(columns={
        "CODE":                   "columna_id",
        "EQUIPMENT_ID":           "equipment_id",
        "TEST_EXECUTION_ID":      "test_execution_id",
        "EXECUTION_DATE":         "execution_date",
        "RULE_NAME":              "rule_name",
        "PARAMETER_ID":           "parameter_id",
        "PARAMETER_NAME":         "parameter_name",
        "ACTUAL_NUMERIC_VALUE":   "actual_value",
        "TARGET_NUMERIC_VALUE":   "target_value",
        "COMPARATOR":             "comparator",
        "PASSED":                 "passed",
    })

    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce")

    df = df[~df["parameter_name"].isin(PARAMETROS_OPERACIONALES)]
    df = df[df["comparator"].notna() & (df["comparator"] != "no_compare")]
    df = df[df["actual_value"].notna() & df["target_value"].notna()]

    # Distancia al umbral normalizada por el valor del umbral.
    # Para mayor/mayor_igual (N >= 8000): (actual − target) / |target|
    # Para menor/menor_igual (Tailing <= 2): (target − actual) / |target|
    abs_target = df["target_value"].abs().replace(0, float("nan"))
    mask_mayor = df["comparator"].isin(COMPARADORES_MAYOR)
    mask_menor = df["comparator"].isin(COMPARADORES_MENOR)

    df["distancia_umbral"] = float("nan")
    df.loc[mask_mayor, "distancia_umbral"] = (
        (df.loc[mask_mayor, "actual_value"] - df.loc[mask_mayor, "target_value"])
        / abs_target[mask_mayor]
    )
    df.loc[mask_menor, "distancia_umbral"] = (
        (df.loc[mask_menor, "target_value"] - df.loc[mask_menor, "actual_value"])
        / abs_target[mask_menor]
    )
    df["distancia_umbral"] = df["distancia_umbral"].astype(float)

    # Eliminar lecturas físicamente imposibles (|distancia| > 100 implica >100x el umbral).
    df = df[df["distancia_umbral"].abs() <= 100]

    df["tipo_columna"]  = df["columna_id"].str.split("-").str[0]
    df["parameter_name"] = df["parameter_name"].replace(UNIFICACION_PARAMETROS)
    df["passed"] = df["passed"].map(
        {True: True, False: False, "True": True, "False": False}
    )

    return df.reset_index(drop=True)


def transformar_inyecciones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula inyecciones acumuladas por columna a lo largo del tiempo.

    Entrada: tabla de tests (una fila por ejecución).
    Salida adicional: inyecciones_acumuladas (cumsum), nro_test (1-based).
    """
    df = df.copy()

    df = df.rename(columns={
        "hplc_column_code":       "columna_id",
        "EXECUTION_DATE":         "execution_date",
        "column_parameter_name":  "parameter_name",
    })

    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce")
    df = df.sort_values(["columna_id", "execution_date"]).reset_index(drop=True)

    df["inyecciones_acumuladas"] = df.groupby("columna_id")["inyecciones"].cumsum()
    df["nro_test"] = df.groupby("columna_id").cumcount() + 1

    return df


def transformar_degradacion_temporal_criterios(
    df_criterios: pd.DataFrame,
    df_inyecciones: pd.DataFrame,
    pivotar: bool = False,
) -> pd.DataFrame:
    """
    Construye la serie temporal de degradación por columna y parámetro SST.

    Une los criterios SST con el contador de inyecciones acumuladas.
    Cada fila = (columna, test, parámetro) con distancia_umbral e inyecciones al momento del test.

    Parámetros
    ----------
    pivotar : si True, una fila por (columna, test) con una columna por parámetro.
              Útil para análisis multivariado.
    """
    inj_key = df_inyecciones[
        ["columna_id", "execution_date", "inyecciones_acumuladas", "nro_test"]
    ].copy()

    merged = df_criterios.merge(inj_key, on=["columna_id", "execution_date"], how="left")

    if pivotar:
        merged = (
            merged.pivot_table(
                index=[
                    "columna_id", "tipo_columna", "execution_date",
                    "inyecciones_acumuladas", "nro_test",
                ],
                columns="parameter_name",
                values="distancia_umbral",
                aggfunc="mean",
            )
            .reset_index()
        )
        merged.columns.name = None

    return merged


# -- Fuente B ------------------------------------------------------------------

def transformar_columnas_fuente_b(
    df: pd.DataFrame, inyecciones_totales: pd.DataFrame
) -> pd.DataFrame:
    """
    Limpia y normaliza el catálogo de columnas de Fuente B.

    Fuente B no provee motivo de baja estructurado: fallo_sst = False (censurada = True).
    """
    df = df.copy()

    df = df.rename(columns={
        "CODE":           "columna_id",
        "RECEPTION_DATE": "fecha_recepcion",
        "ON_USE_DATE":    "fecha_puesta_en_uso",
        "DISPOSAL_DATE":  "fecha_baja",
        "PURPOSE":        "uso_asignado",
        "CLEANING":       "lavado",
        "STORAGE_TYPE":   "almacenamiento",
        "COLUMN_STATE":   "estado",
    })

    for col in ["fecha_recepcion", "fecha_puesta_en_uso", "fecha_baja"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["estado"]       = df["estado"].map(ESTADO_MAP).fillna(df["estado"])
    df["tipo_columna"] = df["columna_id"].str.split("-").str[0]
    df["motivo_baja"]  = pd.NA
    df["fallo_sst"]    = False
    df["censurada"]    = True

    for col in ["nombre", "marca", "nro_serie", "catalogo", "costo",
                "ubicacion", "notas", "fecha_ultimo_uso"]:
        df[col] = pd.NA

    inj = inyecciones_totales.rename(columns={
        "CODE": "columna_id", "TOTAL_INYECCIONES": "total_inyecciones"
    })
    df = df.merge(inj, on="columna_id", how="left")
    df["total_inyecciones"] = df["total_inyecciones"].fillna(0).astype(int)

    df["fuente"] = "Fuente B"
    return df.reset_index(drop=True)


def transformar_inyecciones_fuente_b(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula inyecciones acumuladas por columna para Fuente B."""
    df = df.copy()

    df = df.rename(columns={
        "CODE":                 "columna_id",
        "fecha":                "execution_date",
        "cantidad_inyecciones": "inyecciones",
    })

    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce")
    df = df.sort_values(["columna_id", "execution_date"]).reset_index(drop=True)

    df["inyecciones_acumuladas"] = df.groupby("columna_id")["inyecciones"].cumsum()
    df["nro_test"] = df.groupby("columna_id").cumcount() + 1

    df["fuente"] = "Fuente B"
    return df[["columna_id", "execution_date", "inyecciones",
               "inyecciones_acumuladas", "nro_test", "fuente"]]


# -- Fuente C ------------------------------------------------------------------

def transformar_columnas_fuente_c(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y normaliza el catálogo de columnas de Fuente C.

    Fuente C no tiene criterios SST evaluables. El fallo SST se detecta
    mediante keywords en el campo de notas de baja.
    """
    df = df.copy()

    df = df.rename(columns={
        "CODE":           "columna_id",
        "RECEPTION_DATE": "fecha_recepcion",
        "ON_USE_DATE":    "fecha_puesta_en_uso",
        "DISPOSAL_DATE":  "fecha_baja",
        "PURPOSE":        "uso_asignado",
        "CLEANING":       "lavado",
        "STORAGE_TYPE":   "almacenamiento",
        "COLUMN_STATE":   "estado",
        "DISPOSALNOTES":  "motivo_baja",
    })

    for col in ["fecha_recepcion", "fecha_puesta_en_uso", "fecha_baja"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["estado"]       = df["estado"].map(ESTADO_MAP).fillna(df["estado"])
    df["tipo_columna"] = df["columna_id"].str.split("-").str[0]
    df["fallo_sst"]    = _detectar_fallo_sst_por_notas(df["motivo_baja"])
    df["censurada"]    = ~df["fallo_sst"]

    for col in ["nombre", "marca", "nro_serie", "catalogo", "costo",
                "ubicacion", "notas", "fecha_ultimo_uso", "total_inyecciones"]:
        df[col] = pd.NA

    df["fuente"] = "Fuente C"
    return df.reset_index(drop=True)


def transformar_inyecciones_fuente_c(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula inyecciones acumuladas por columna para Fuente C.

    Filtra solo las filas del parámetro 'inyecciones' (Fuente C registra
    múltiples parámetros por test incluyendo presión).
    """
    df = df.copy()

    df = df[df["nombre_parametro"].str.lower() == "inyecciones"].copy()

    df = df.rename(columns={
        "CODE":              "columna_id",
        "fecha_test":        "execution_date",
        "valor_numerico":    "inyecciones",
        "TEST_EXECUTION_ID": "test_execution_id",
    })

    df["execution_date"] = pd.to_datetime(df["execution_date"], errors="coerce")
    df = df.sort_values(["columna_id", "execution_date"]).reset_index(drop=True)

    df["inyecciones_acumuladas"] = df.groupby("columna_id")["inyecciones"].cumsum()
    df["nro_test"] = df.groupby("columna_id").cumcount() + 1

    df["fuente"] = "Fuente C"
    return df[["columna_id", "execution_date", "inyecciones",
               "inyecciones_acumuladas", "nro_test", "fuente"]]
