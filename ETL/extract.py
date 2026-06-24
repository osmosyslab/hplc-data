"""
Carga de los archivos CSV raw por fuente.

Cada fuente tiene su propia estructura de archivos y encoding.
Las rutas son relativas al directorio data/ del proyecto.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# -- Fuente A -----------------------------------------------------------------
columnas_hplc = pd.read_csv(DATA_DIR / "ColumnasHplc.csv", encoding="utf-8")

# CriteriosXTest.csv puede tener strings sin cerrar en alguna fila
criteriosXtest = pd.read_csv(
    DATA_DIR / "CriteriosXTest.csv",
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip",
)

inyeccionesXtest = pd.read_csv(DATA_DIR / "Resultados_inyeccionesXtest.csv", encoding="utf-8")

# -- Fuente B (misma estructura que Fuente A, columnas en inglés) --------------
columnas_aspen = pd.read_csv(DATA_DIR / "ColumnasHPLC_ASPEN.csv", encoding="latin-1")

criterios_aspen = pd.read_csv(
    DATA_DIR / "Resultados_tests_ASPEN.csv",
    encoding="latin-1",
    engine="python",
    on_bad_lines="skip",
)

# Solo contiene CODE + TOTAL_INYECCIONES (sin registros por test)
inyecciones_totales_aspen = pd.read_csv(DATA_DIR / "InyeccionesXColumnasASPEN.csv", encoding="utf-8")

# Trayectoria de inyecciones por test y fecha (CODE, fecha, cantidad_inyecciones)
trayectoria_inyecciones_aspen = pd.read_csv(DATA_DIR / "TrayectoriaInyeccionesAspen.csv", encoding="utf-8")

# -- Fuente C (estructura diferente, sin criterios SST evaluables) -------------
columnas_roche = pd.read_csv(DATA_DIR / "ColumnasHPLC_Roche.csv", encoding="utf-8")

# Contiene inyecciones y presión por test; sin criterios SST evaluables
tests_roche = pd.read_csv(DATA_DIR / "Resultados_tests_Roche.csv", encoding="latin-1")

# -- Especificaciones de columnas (por fuente) ---------------------------------
especificaciones_aspen       = pd.read_csv(DATA_DIR / "Especificaciones_Columnas_Aspen.csv",      encoding="latin-1")
especificaciones_roche       = pd.read_csv(DATA_DIR / "Especificaciones_Columnas_Roche.csv",      encoding="latin-1")
especificaciones_bayer_lerma = pd.read_csv(DATA_DIR / "Especificaciones_Columnas_BayerLerma.csv", encoding="latin-1")
