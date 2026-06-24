"""
Modelo de supervivencia para columnas HPLC de tipo C-18.
Subconjunto de survival.py restringido a tipo_relleno == "C18".

Pipeline
--------
1. cargar_datos          -> DataFrames desde model_data/  (reusa survival.py)
2. preparar_survival_c18 -> dataset filtrado solo a columnas C-18,
                            enriquecido con variables fisicas de specs
3. kaplan_meier          -> curvas KM globales y por fuente  (reusa survival.py)
4. cox_ph                -> hazard ratios con covariables C-18  (reusa survival.py)
5. lognormal_aft         -> modelo Log-Normal AFT  (reusa survival.py)
6. evaluar_modelo        -> C-index OOS k-fold CV  (reusa survival.py)
7. predecir_vida_util    -> percentiles P25/P50/P75 de vida util  (reusa survival.py)

Covariables utiles en C-18 (tipo_relleno es constante, no aporta):
  - fuente: laboratorio de origen (Bayer Lerma, ASPEN, Roche).
            Unica covariable viable: tipo_columna genera overfitting severo
            (EPV=0.16 con 12 eventos y 74 tipos), y las variables fisicas
            (longitud, tamanio_particula) no muestran señal estadistica (p>0.2).
"""

from pathlib import Path

import pandas as pd

from model.survival import (
    cargar_datos,
    kaplan_meier,
    cox_ph,
    weibull_aft,
    lognormal_aft,
    generalized_gamma_aft,
    comparar_modelos,
    verificar_supuesto_aft,
    evaluar_modelo,
    buscar_penalizer,
    predecir_vida_util,
    describir_survival,
    plot_survival_data,
    plot_kaplan_meier,
    _codificar_covariables,
)

COVARIABLES_DEFAULT = ["fuente"]

# Columnas fisicas deseadas de Especificaciones_columnas.csv.
# Se buscan por substring para no depender del encoding exacto del caracter ñ.
# El valor es el nombre destino en el dataset de supervivencia.
_SPECS_FISICAS_SUBSTR = {
    "longitud":   "longitud",
    "diametro":   "diametro",
    "particula":  "tamanio_particula",   # matchea "tamaño_particula" o "tamano_particula"
}


def _renombrar_specs_fisicas(specs: pd.DataFrame) -> dict[str, str]:
    """Construye el mapa {nombre_original: nombre_destino} buscando por substring."""
    rename = {}
    for substr, destino in _SPECS_FISICAS_SUBSTR.items():
        matches = [c for c in specs.columns if substr in c.lower()]
        if matches:
            rename[matches[0]] = destino
    return rename


# -- 2. Preparacion del dataset filtrado a C-18 --------------------------------

def preparar_survival_c18(
    columnas: pd.DataFrame,
    inyecciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de supervivencia restringido a columnas C-18,
    enriquecido con variables fisicas de Especificaciones_columnas.csv.

    Campos resultantes
    ------------------
    columna_id         : identificador de la columna
    tiempo             : inyecciones acumuladas al evento o al ultimo registro
    evento             : 1 = fallo SST observado, 0 = censurada
    fuente             : Bayer Lerma / ASPEN / Roche
    tipo_columna       : codigo de columna (ej. C001, C186, C218 ...)
    tipo_relleno       : siempre "C18" en este dataset
    longitud           : longitud de la columna en mm
    diametro           : diametro interno en mm
    tamanio_particula  : tamaño de particula en µm (distingue UPLC de HPLC)

    Notes
    -----
    - Se excluyen columnas sin match en Especificaciones_columnas.csv.
    - tipo_relleno es constante (C18); no usarla como covariable.
    - Las variables fisicas son NaN cuando el codigo de columna no tiene
      entrada en specs; el modelo las maneja via imputacion en _codificar_covariables
      si se activa, o se pueden descartar esas filas antes de ajustar.
    """
    from model.survival import preparar_survival, SPECS_PATH

    if not SPECS_PATH.exists():
        raise FileNotFoundError(
            f"Especificaciones_columnas.csv no encontrado en {SPECS_PATH}. "
            "Es necesario para identificar columnas C-18."
        )

    survival = preparar_survival(columnas, inyecciones)

    if "tipo_relleno" not in survival.columns:
        raise ValueError(
            "La columna tipo_relleno no fue generada. "
            "Verificar que Especificaciones_columnas.csv contiene la columna tipo_relleno."
        )

    c18 = survival[survival["tipo_relleno"] == "C18"].reset_index(drop=True)

    if c18.empty:
        raise ValueError(
            "No se encontraron columnas con tipo_relleno == 'C18'. "
            "Verificar clasificacion en Especificaciones_columnas.csv."
        )

    # -- Enriquecer con variables fisicas de specs --------------------------------
    specs = pd.read_csv(SPECS_PATH, encoding="utf-8-sig")

    rename_map = _renombrar_specs_fisicas(specs)
    cols_originales = list(rename_map.keys())
    specs_fisicas = (
        specs[["code", "fuente"] + cols_originales]
        .copy()
        .rename(columns=rename_map)
    )
    specs_fisicas["fuente"] = specs_fisicas["fuente"].str.strip()

    fuente_norm = c18["fuente"].str.strip()
    c18 = c18.assign(_fuente_norm=fuente_norm)
    specs_fisicas = specs_fisicas.rename(columns={"code": "tipo_columna", "fuente": "_fuente_norm"})

    c18 = (
        c18
        .merge(specs_fisicas, on=["tipo_columna", "_fuente_norm"], how="left")
        .drop(columns="_fuente_norm")
    )

    return c18.reset_index(drop=True)


# -- Pipeline completo ---------------------------------------------------------

def pipeline_c18(
    covariables: list[str] = COVARIABLES_DEFAULT,
    penalizer: float = 0.1,
    k_cv: int = 5,
    summary: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de supervivencia para columnas C-18.

    Pasos
    -----
    1. Carga datos
    2. Prepara dataset C-18
    3. Describe el dataset
    4. Ajusta Log-Normal AFT (modelo principal)
    5. Evalua con k-fold CV
    6. Retorna modelo y dataset

    Parameters
    ----------
    covariables : covariables a incluir. Default: ["fuente"].
    penalizer   : regularizacion L2. Default: 0.1.
    k_cv        : folds para cross-validation. Default: 5.
    summary     : si True, imprime resumen en cada paso.

    Returns
    -------
    dict con claves:
      survival_c18 : pd.DataFrame -- dataset filtrado a C-18
      modelo       : LogNormalAFTFitter ajustado
      metricas     : dict con C-index OOS, AIC, log-likelihood
    """
    columnas, _, inyecciones, _ = cargar_datos()
    survival_c18 = preparar_survival_c18(columnas, inyecciones)

    if summary:
        describir_survival(survival_c18)

    modelo = lognormal_aft(
        survival_c18,
        covariables=covariables,
        penalizer=penalizer,
        summary=summary,
    )

    metricas = evaluar_modelo(
        modelo,
        survival_c18,
        covariables=covariables,
        k=k_cv,
        penalizer=penalizer,
    )

    return {
        "survival_c18": survival_c18,
        "modelo": modelo,
        "metricas": metricas,
    }
