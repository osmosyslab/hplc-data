"""
Persistencia de los DataFrames procesados en data/proc/.

Estructura de salida
--------------------
data/proc/
    columnas.csv        -- catálogo de columnas (todas las fuentes)
    criterios.csv       -- evaluaciones SST
    inyecciones.csv     -- tests con conteos acumulados
    degradacion.csv     -- serie temporal degradación × inyecciones
    Especificaciones_columnas.csv
    model_data/         -- subconjunto filtrado para modelado (> 1 inyección)
"""
from pathlib import Path
import pandas as pd

PROC_DIR  = Path(__file__).parent.parent / "data" / "proc"
MODEL_DIR = PROC_DIR / "model_data"


def guardar_datos(columnas, criterios, inyecciones, degradacion):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    columnas.to_csv(PROC_DIR / "columnas.csv", index=False)
    criterios.to_csv(PROC_DIR / "criterios.csv", index=False)
    inyecciones.to_csv(PROC_DIR / "inyecciones.csv", index=False)
    degradacion.to_csv(PROC_DIR / "degradacion.csv", index=False)
    print(f"  Datos guardados en {PROC_DIR}")


def cargar_datos():
    return (
        pd.read_csv(PROC_DIR / "columnas.csv"),
        pd.read_csv(PROC_DIR / "criterios.csv"),
        pd.read_csv(PROC_DIR / "inyecciones.csv"),
        pd.read_csv(PROC_DIR / "degradacion.csv"),
    )


def guardar_model_data(columnas: pd.DataFrame, criterios: pd.DataFrame,
                       inyecciones: pd.DataFrame, degradacion: pd.DataFrame):
    """
    Guarda en data/proc/model_data/ los mismos cuatro archivos que proc/,
    filtrados a las columnas con más de 1 inyección acumulada registrada.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Filtrar por (columna_id, fuente) para evitar que columnas de una fuente
    # hereden registros de inyecciones de otra fuente con el mismo código.
    pares_validos = (
        inyecciones[inyecciones["inyecciones_acumuladas"] > 1][["columna_id", "fuente"]]
        .drop_duplicates()
        .assign(_keep=True)
    )

    def _filtrar(df):
        return (
            df.merge(pares_validos, on=["columna_id", "fuente"], how="left")
            .query("_keep == True")
            .drop(columns="_keep")
            .reset_index(drop=True)
        )

    col_m  = _filtrar(columnas)
    crit_m = _filtrar(criterios)
    inj_m  = _filtrar(inyecciones)
    deg_m  = _filtrar(degradacion)

    col_m.to_csv(MODEL_DIR / "columnas.csv", index=False)
    crit_m.to_csv(MODEL_DIR / "criterios.csv", index=False)
    inj_m.to_csv(MODEL_DIR / "inyecciones.csv", index=False)
    deg_m.to_csv(MODEL_DIR / "degradacion.csv", index=False)

    print(f"  model_data/ -> {MODEL_DIR}")
    print(f"    columnas   : {len(col_m)} ({col_m['fallo_sst'].sum()} con fallo SST)")
    print(f"    criterios  : {len(crit_m)}")
    print(f"    inyecciones: {len(inj_m)}")
    print(f"    degradacion: {len(deg_m)}")


def guardar_especificaciones(especificaciones: pd.DataFrame):
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    especificaciones.to_csv(PROC_DIR / "Especificaciones_columnas.csv", index=False)
    print(f"  Especificaciones guardadas en {PROC_DIR / 'Especificaciones_columnas.csv'} ({len(especificaciones)} registros)")


def datos_procesados_existen():
    return all(
        (PROC_DIR / f).exists()
        for f in ["columnas.csv", "criterios.csv", "inyecciones.csv", "degradacion.csv"]
    )
