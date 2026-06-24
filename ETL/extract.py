"""
Carga de los archivos CSV raw por fuente.

Cada fuente tiene su propia estructura de archivos y encoding.
Reemplazar los nombres de archivo con los de su propia fuente de datos.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# -- Fuente A -----------------------------------------------------------------
columnas_a = pd.read_csv(DATA_DIR / "columnas_fuente_a.csv", encoding="utf-8")

# Puede tener strings sin cerrar en alguna fila
criterios_a = pd.read_csv(
    DATA_DIR / "criterios_fuente_a.csv",
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip",
)

inyecciones_a = pd.read_csv(DATA_DIR / "inyecciones_fuente_a.csv", encoding="utf-8")

# -- Fuente B (misma estructura que Fuente A, columnas en inglés) --------------
columnas_b = pd.read_csv(DATA_DIR / "columnas_fuente_b.csv", encoding="latin-1")

criterios_b = pd.read_csv(
    DATA_DIR / "criterios_fuente_b.csv",
    encoding="latin-1",
    engine="python",
    on_bad_lines="skip",
)

# Solo contiene CODE + TOTAL_INYECCIONES (sin registros por test)
inyecciones_totales_b = pd.read_csv(DATA_DIR / "inyecciones_totales_fuente_b.csv", encoding="utf-8")

# Trayectoria de inyecciones por test y fecha (CODE, fecha, cantidad_inyecciones)
trayectoria_inyecciones_b = pd.read_csv(DATA_DIR / "trayectoria_inyecciones_fuente_b.csv", encoding="utf-8")

# -- Fuente C (estructura diferente, sin criterios SST evaluables) -------------
columnas_c = pd.read_csv(DATA_DIR / "columnas_fuente_c.csv", encoding="utf-8")

# Contiene inyecciones y presión por test; sin criterios SST evaluables
tests_c = pd.read_csv(DATA_DIR / "tests_fuente_c.csv", encoding="latin-1")

# -- Especificaciones de columnas (por fuente) ---------------------------------
especificaciones_a = pd.read_csv(DATA_DIR / "especificaciones_fuente_a.csv", encoding="latin-1")
especificaciones_b = pd.read_csv(DATA_DIR / "especificaciones_fuente_b.csv", encoding="latin-1")
especificaciones_c = pd.read_csv(DATA_DIR / "especificaciones_fuente_c.csv", encoding="latin-1")
