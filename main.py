"""
Orquestador del pipeline completo: ETL → Modelos de supervivencia → Clasificación.

Uso
---
Ajustar las flags en __main__ y ejecutar:

    python main.py

Si los datos procesados ya existen en data/proc/, el ETL se omite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ETL.extract import (
    columnas_a, criterios_a, inyecciones_a,
    columnas_b, criterios_b, inyecciones_totales_b, trayectoria_inyecciones_b,
    columnas_c, tests_c,
    especificaciones_a, especificaciones_b, especificaciones_c,
)
from ETL.transform import (
    transformar_columnas_hplc,
    transformar_resultados_criterios,
    transformar_inyecciones,
    transformar_degradacion_temporal_criterios,
    transformar_columnas_fuente_b,
    transformar_inyecciones_fuente_b,
    transformar_columnas_fuente_c,
    transformar_inyecciones_fuente_c,
)
from ETL.load import guardar_datos, guardar_model_data, guardar_especificaciones, cargar_datos, datos_procesados_existen
import pandas as pd


def run_etl():
    # -- Fuente A -------------------------------------------------------------
    col_a        = transformar_columnas_hplc(columnas_a)
    col_a["fuente"]   = "Fuente A"
    crit_a       = transformar_resultados_criterios(criterios_a)
    crit_a["fuente"]  = "Fuente A"
    inj_a        = transformar_inyecciones(inyecciones_a)
    inj_a["fuente"]   = "Fuente A"

    # -- Fuente B -------------------------------------------------------------
    col_b   = transformar_columnas_fuente_b(columnas_b, inyecciones_totales_b)
    crit_b  = transformar_resultados_criterios(criterios_b)
    crit_b["fuente"] = "Fuente B"
    inj_b   = transformar_inyecciones_fuente_b(trayectoria_inyecciones_b)

    # Códigos que aparecen en Fuente A y en Fuente B son columnas físicamente
    # distintas en sitios distintos. Se desambiguan añadiendo sufijo "_B"
    # en todas las tablas de Fuente B para evitar colisiones de columna_id.
    overlap_a_b = set(col_a["columna_id"]) & set(col_b["columna_id"])
    if overlap_a_b:
        def _sufijo_b(df, col="columna_id"):
            df = df.copy()
            df.loc[df[col].isin(overlap_a_b), col] += "_B"
            return df
        col_b   = _sufijo_b(col_b)
        crit_b  = _sufijo_b(crit_b)
        inj_b   = _sufijo_b(inj_b)

    # -- Fuente C -------------------------------------------------------------
    col_c = transformar_columnas_fuente_c(columnas_c)
    inj_c = transformar_inyecciones_fuente_c(tests_c)

    # Poblar total_inyecciones desde el máximo acumulado por columna
    totales_c = (
        inj_c.groupby("columna_id")["inyecciones_acumuladas"]
        .max()
        .reset_index()
        .rename(columns={"inyecciones_acumuladas": "total_inyecciones"})
    )
    col_c = (
        col_c.drop(columns="total_inyecciones")
        .merge(totales_c, on="columna_id", how="left")
    )

    # -- Merge ----------------------------------------------------------------
    columnas    = pd.concat([col_a, col_b, col_c], ignore_index=True)
    criterios   = pd.concat([crit_a, crit_b], ignore_index=True)

    inj_cols = ["columna_id", "execution_date", "inyecciones", "inyecciones_acumuladas", "nro_test", "fuente"]
    inyecciones = pd.concat(
        [inj_a[inj_cols], inj_b[inj_cols], inj_c[inj_cols]],
        ignore_index=True,
    )

    degradacion = transformar_degradacion_temporal_criterios(criterios, inyecciones)

    # -- Resumen --------------------------------------------------------------
    print("=== Columnas HPLC ===")
    print(f"  {len(columnas)} columnas | {columnas['fallo_sst'].sum()} con fallo SST")
    for fuente, grp in columnas.groupby("fuente"):
        print(f"  [{fuente}] {len(grp)} columnas | {grp['fallo_sst'].sum()} fallos")

    print("\n=== Criterios SST ===")
    print(f"  {len(criterios)} registros | {criterios['columna_id'].nunique()} columnas")
    for fuente, grp in criterios.groupby("fuente"):
        print(f"  [{fuente}] {len(grp)} registros | {(grp['passed'] == False).sum()} fallos")

    print("\n=== Inyecciones ===")
    print(f"  {len(inyecciones)} tests | {inyecciones['columna_id'].nunique()} columnas")
    for fuente, grp in inyecciones.groupby("fuente"):
        print(f"  [{fuente}] {len(grp)} tests | {grp['columna_id'].nunique()} columnas")

    print("\n=== Serie temporal de degradacion ===")
    print(f"  {len(degradacion)} filas (columna x test x parámetro)")
    print(f"  distancia_umbral < 0 (fallos): {(degradacion['distancia_umbral'] < 0).sum()}")

    # -- Especificaciones de columnas -----------------------------------------
    especificaciones_a["fuente"] = "Fuente A"
    especificaciones_b["fuente"] = "Fuente B"
    especificaciones_c["fuente"] = "Fuente C"
    especificaciones = pd.concat(
        [especificaciones_a, especificaciones_b, especificaciones_c],
        ignore_index=True,
    )

    guardar_datos(columnas, criterios, inyecciones, degradacion)
    guardar_model_data(columnas, criterios, inyecciones, degradacion)
    guardar_especificaciones(especificaciones)

    return columnas, criterios, inyecciones, degradacion


def obtener_datos():
    """Devuelve los DataFrames desde proc/ si existen; si no, ejecuta el ETL."""
    if datos_procesados_existen():
        print("Cargando datos desde data/proc/...")
        return cargar_datos()
    print("Datos procesados no encontrados, ejecutando ETL...")
    return run_etl()


def run_survival():
    from model.survival import (
        cargar_datos, preparar_survival, describir_survival,
        plot_survival_data, plot_kaplan_meier,
        weibull_aft, lognormal_aft, generalized_gamma_aft,
        comparar_modelos,
        verificar_supuesto_aft,
        evaluar_modelo,
        predecir_vida_util,
    )

    COVARIABLES = ["fuente", "tipo_relleno"]
    PENALIZER   = 0.1
    # Ejemplos de columnas para demostrar la predicción de vida útil
    COLUMNAS    = ["C218-1", "C124-27", "C054-1"]

    # -- 1. Datos -------------------------------------------------------------
    columnas, criterios, inyecciones, degradacion = cargar_datos()
    survival = preparar_survival(columnas, inyecciones)

    # -- 2. Exploración -------------------------------------------------------
    describir_survival(survival)
    plot_survival_data(survival)
    plot_kaplan_meier(survival)

    # -- 3. Diagnóstico de distribución (Generalized Gamma) -------------------
    generalized_gamma_aft(survival, summary=True)

    # -- 4. Ajuste y comparación de modelos AFT -------------------------------
    weibull_aft(survival, COVARIABLES, penalizer=PENALIZER, summary=True)
    modelo_ln = lognormal_aft(survival, COVARIABLES, penalizer=PENALIZER, summary=True)
    comparar_modelos(survival, COVARIABLES, penalizer=PENALIZER)

    # -- 5. Verificación del supuesto AFT -------------------------------------
    verificar_supuesto_aft(modelo_ln, survival, COVARIABLES)

    # -- 6. Evaluación out-of-sample ------------------------------------------
    evaluar_modelo(modelo_ln, survival, COVARIABLES, penalizer=PENALIZER)

    # -- 7. Predicción de vida útil (P25 / P50 / P75) ------------------------
    etiquetas = {
        0.25: "P25 (optimista) ",
        0.50: "P50 (mediana)   ",
        0.75: "P75 (conservador)",
    }
    print(f"\n{'='*60}")
    print(f"  Predicción de vida útil — Log-Normal AFT")
    print(f"  (P25 = optimista, P50 = mediana, P75 = conservador)")
    print(f"{'='*60}")
    for col_id in COLUMNAS:
        pred = predecir_vida_util(modelo_ln, col_id, survival_df=survival)
        fila = survival[survival["columna_id"] == col_id].iloc[0]
        print(
            f"\n  {col_id}  [{fila['fuente']} / {fila.get('tipo_relleno', '?')}]"
            f"  evento={int(fila['evento'])}  t_obs={int(fila['tiempo'])}"
        )
        for p, inyecciones_est in pred.items():
            print(f"    {etiquetas[p]}: {int(inyecciones_est):>8,}")
    print(f"{'='*60}")


def run_clasificacion() -> None:
    from model.clasificador import visualizar_clusters, plot_clusters_vs_inyecciones
    visualizar_clusters()
    plot_clusters_vs_inyecciones()


if __name__ == "__main__":
    CORRER_SURVIVAL      = True
    CORRER_CLASIFICACION = False

    columnas, criterios, inyecciones, degradacion = obtener_datos()
    if CORRER_SURVIVAL:
        run_survival()
    if CORRER_CLASIFICACION:
        run_clasificacion()
