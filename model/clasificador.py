"""Modelo de clasificacion de columnas por su tipo bajo varios
 modelos de aprendizaje no supervisados.

"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

MODEL_DATA_DIR = Path(__file__).parent.parent / "data" / "proc"


def cargar_datos() -> pd.DataFrame:
    ruta = MODEL_DATA_DIR / "Especificaciones_columnas.csv"
    return pd.read_csv(ruta, encoding="utf-8-sig")

def escalar_datos_numericos() -> np.ndarray:
    df = cargar_datos()
    features = df[["longitud", "diametro", "t_particula"]].to_numpy()
    features = SimpleImputer(strategy="mean").fit_transform(features)
    return StandardScaler().fit_transform(features)

_TIPOS_RELLENO = {
    "C18":           ["c18", "ods", "octadecyl", "octadecil", "rp-18", "rp18", "rp 18",
                      "lc-18", "lc18", "c18-db", "betabasic-18", "ascentis express c-18",
                      "xtimate", "gemini", "t3", "cortes uplc"],
    "C8":            ["c8", "c08", "octyl", "octil", "rp-8", "rp8", "rp 8",
                      "lc-8", "lc8", "shield rp8", "octilico"],
    "Phenyl":        ["phenyl", "fenil"],
    "CN":            ["cn", "cyano", "ciano", "nitrile"],
    "NH2":           ["nh2", "amino"],
    "HILIC":         ["hilic", "amide", "amida", "zic", "diol", "hidroxilo"],
    "Chiral":        ["chiral", "chiralpak", "chiralcel", "chiracel", "chiradex",
                      "celulosa", "polisacarido", "polisacárido", "vancomycin"],
    "Silica":        ["silica", "porasil", "hibar", "lichrospher", "lichrosorb",
                      "nucleosil", "spherisorb", "ultrasphere", "ultremex", "rx-sil",
                      "completamente porosa"],
    "AQ":            ["aq", "aqueous", "sb-aq", "rp-aq"],
    "IonExchange":   ["scx", "wcx", "iex", "deae", "ionpac", "carbopac", "ion pac",
                      "bioproiex", "bioproi", "propac", "mabpac", "acid", "sulfon",
                      "sulfonado", "cation", "anion", "hidroge", "calcio", "plomo"],
    "SizeExclusion": ["sw", "tsk g", "superose", "size exclusion", "tskg", "tsk gel"],
    "GC":            ["db-", "db1", "rtx", "rxi", "wax", "hp-", "zb-", "ffap",
                      "stabilwax", "shrx", "ms", "624"],
}

def _clasificar_relleno(texto: str) -> str:
    t = texto.lower()
    for tipo, keywords in _TIPOS_RELLENO.items():
        if any(kw in t for kw in keywords):
            return tipo
    return "Otro"


def escalar_datos_categoricos() -> np.ndarray:
    df = cargar_datos()
    resto = df[["nombre_marca", "code"]].fillna("").agg(" ".join, axis=1)

    tipo_relleno = df["relleno"].fillna("").apply(_clasificar_relleno)
    dummies = pd.get_dummies(tipo_relleno).to_numpy().astype(float) * 2.0

    matriz_resto = TfidfVectorizer().fit_transform(resto).toarray()

    return np.hstack([matriz_resto, dummies])


def encontrar_codo(inertias: list[float]) -> int:
    # Distancia de cada punto a la recta que une el primero y el último
    n = len(inertias)
    p1 = np.array([0, inertias[0]])
    p2 = np.array([n - 1, inertias[-1]])
    recta = p2 - p1
    distancias = [
        abs(np.cross(recta, p1 - np.array([i, inertias[i]]))) / np.linalg.norm(recta)
        for i in range(n)
    ]
    return int(np.argmax(distancias)) + 2 


def clasificar() -> pd.DataFrame:
    df = cargar_datos()
    df["tipo_relleno"] = df["relleno"].fillna("").apply(_clasificar_relleno)
    df.to_csv(MODEL_DATA_DIR / "Especificaciones_columnas.csv", index=False, encoding="utf-8-sig")
    return df


def visualizar_clusters() -> None:
    df = clasificar()
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))


def plot_codo(k_max: int = 10) -> None:
    numericos = escalar_datos_numericos()
    categoricos = escalar_datos_categoricos()
    features = StandardScaler().fit_transform(np.hstack([numericos, categoricos]))

    ks = range(2, k_max + 1)
    inertias = [
        KMeans(n_clusters=k, random_state=42).fit(features).inertia_
        for k in ks
    ]
    k_optimo = encontrar_codo(inertias)

    _, ax = plt.subplots()
    ax.plot(list(ks), inertias, marker="o", label="Inercia")
    ax.axvline(k_optimo, color="red", linestyle="--", label=f"k óptimo = {k_optimo}")
    ax.set_xlabel("Número de clusters (k)")
    ax.set_ylabel("Inercia")
    ax.set_title("Método del codo")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_clusters_vs_inyecciones() -> None:
    columnas   = pd.read_csv(MODEL_DATA_DIR / "columnas.csv")
    inyecciones = pd.read_csv(MODEL_DATA_DIR / "inyecciones.csv")

    total_inj = (
        inyecciones.groupby("columna_id")["inyecciones_acumuladas"]
        .max()
        .reset_index()
        .rename(columns={"inyecciones_acumuladas": "total_inyecciones"})
    )

    especificaciones = clasificar()[["code", "tipo_relleno"]]
    df = (
        columnas[["columna_id", "tipo_columna"]]
        .merge(especificaciones, left_on="tipo_columna", right_on="code")
        .merge(total_inj, on="columna_id")
    )
    df = df[df["total_inyecciones"] > 10]

    tipos = sorted(df["tipo_relleno"].unique())
    cmap = plt.colormaps["tab10"]
    colores = {t: cmap(i / len(tipos)) for i, t in enumerate(tipos)}

    _, ax = plt.subplots(figsize=(14, 5))
    for cluster, grupo in df.groupby("tipo_relleno"):
        ax.scatter(
            grupo["columna_id"],
            grupo["total_inyecciones"],
            label=cluster,
            color=colores[cluster],
            alpha=0.8,
        )

    titulo = "Inyecciones por columna cromatográfica según tipo de relleno"
    ax.set_xlabel("Columna cromatográfica")
    ax.set_ylabel("Total de inyecciones")
    ax.set_title(titulo)
    ax.set_xticks([])
    ax.legend(title="Tipo de relleno")
    plt.tight_layout()

    plots_dir = Path(__file__).parent.parent / "plots"
    plt.savefig(plots_dir / "inyecciones_por_tipo_relleno.png", dpi=150)
    plt.show()


def evaluar_clusters(k_max: int = 10) -> None:
    numericos   = escalar_datos_numericos()
    categoricos = escalar_datos_categoricos()
    features    = StandardScaler().fit_transform(np.hstack([numericos, categoricos]))

    print(f"{'k':>4} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20} {'Tamaños clusters'}")
    print("-" * 80)
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(features)
        sil    = silhouette_score(features, labels)
        db     = davies_bouldin_score(features, labels)
        ch     = calinski_harabasz_score(features, labels)
        sizes  = sorted(np.bincount(labels), reverse=True)
        print(f"{k:>4} {sil:>12.3f} {db:>16.3f} {ch:>20.1f} {sizes}")
