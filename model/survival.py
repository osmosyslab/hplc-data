"""
Modelo de supervivencia para columnas HPLC.
Objetivo: estimar cuantas inyecciones va a durar una columna.

Pipeline
--------
1. cargar_datos           -> DataFrames desde model_data/
2. preparar_survival      -> dataset (columna_id, tiempo, evento, covariables)
3. kaplan_meier           -> curvas de supervivencia no parametricas por grupo
4. cox_ph                 -> hazard ratios de covariables (tipo_relleno, fuente)
5. weibull_aft            -> modelo parametrico Weibull AFT
   lognormal_aft          -> modelo parametrico Log-Normal AFT (modelo principal)
   generalized_gamma_aft  -> diagnostico de seleccion de distribucion
   comparar_modelos       -> comparacion Weibull vs Log-Normal (AIC, C-index CV)
6. verificar_supuesto_aft -> diagnostico grafico del supuesto AFT
7. evaluar_modelo         -> C-index out-of-sample (k-fold CV)
   buscar_penalizer       -> seleccion de regularizacion por CV
8. predecir_vida_util     -> percentiles P25/P50/P75 de vida util para una columna

Modelo elegido: Log-Normal AFT.
  - El hazard log-normal es no monotono (sube y baja), consistente con el
    patron de fallos tempranos observado en los datos.
  - El modelo de cura (Mixture Cure) fue evaluado y descartado: con 8.6% de
    eventos y censuramiento administrativo, la fraccion "curada" es
    matematicamente indistinguible de columnas que simplemente no se observaron
    el tiempo suficiente. No mejora la log-lik OOS respecto del AFT estandar.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm as scipy_norm
from lifelines import (
    CoxPHFitter,
    WeibullAFTFitter,
    LogNormalAFTFitter,
    KaplanMeierFitter,
    GeneralizedGammaFitter,
)
from lifelines.statistics import logrank_test, multivariate_logrank_test

MODEL_DATA_DIR = Path(__file__).parent.parent / "data" / "proc" / "model_data"
SPECS_PATH     = Path(__file__).parent.parent / "data" / "proc" / "Especificaciones_columnas.csv"


# -- 1. Carga -----------------------------------------------------------------

def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Lee los cuatro CSVs de model_data/ y los devuelve como DataFrames.

    Returns
    -------
    columnas, criterios, inyecciones, degradacion
    """
    columnas    = pd.read_csv(MODEL_DATA_DIR / "columnas.csv")
    criterios   = pd.read_csv(MODEL_DATA_DIR / "criterios.csv")
    inyecciones = pd.read_csv(MODEL_DATA_DIR / "inyecciones.csv",
                              parse_dates=["execution_date"])
    degradacion = pd.read_csv(MODEL_DATA_DIR / "degradacion.csv")
    return columnas, criterios, inyecciones, degradacion


# -- 2. Preparacion del dataset de supervivencia ------------------------------

def preparar_survival(
    columnas: pd.DataFrame,
    inyecciones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construye el dataset de supervivencia: una fila por columna.

    Campos resultantes
    ------------------
    columna_id   : identificador de la columna
    tiempo       : inyecciones acumuladas al evento o al ultimo registro
    evento       : 1 = fallo SST observado, 0 = censurada
    fuente       : Bayer Lerma / ASPEN / Roche
    tipo_columna : prefijo del codigo (C124, COL014, ...)
    tipo_relleno : fase estacionaria agrupada (C18, C8, IonExchange, ...)
                   desde Especificaciones_columnas.csv; NaN si no hay match.

    Notes
    -----
    - Tiempo al evento: maximo de inyecciones_acumuladas registrado para esa columna.
    - Se excluyen columnas con tiempo <= 1 (ya filtradas en model_data/).
    - El merge con specs usa (tipo_columna, fuente) para evitar duplicados entre
      fuentes que comparten el mismo codigo de columna con specs distintas.
    """
    tiempo = (
        inyecciones[inyecciones["inyecciones_acumuladas"] > 1]
        .groupby("columna_id")["inyecciones_acumuladas"]
        .max()
        .reset_index()
        .rename(columns={"inyecciones_acumuladas": "tiempo"})
    )

    survival = (
        columnas[["columna_id", "fuente", "tipo_columna", "fallo_sst"]]
        .merge(tiempo, on="columna_id", how="inner")
        .rename(columns={"fallo_sst": "evento"})
    )
    survival["evento"] = survival["evento"].astype(int)

    # -- Enriquecer con tipo_relleno desde especificaciones -------------------
    if SPECS_PATH.exists():
        specs = (
            pd.read_csv(SPECS_PATH)[["code", "fuente", "tipo_relleno"]]
            .rename(columns={"code": "tipo_columna"})
        )
        survival = survival.merge(specs, on=["tipo_columna", "fuente"], how="left")

    return survival.reset_index(drop=True)


# -- Helpers internos ---------------------------------------------------------

def _agrupar_categorias_raras(
    df: pd.DataFrame,
    col: str,
    min_obs: int = 5,
    etiqueta: str = "Otro",
) -> pd.DataFrame:
    """
    Agrupa en 'etiqueta' las categorias de 'col' con menos de min_obs observaciones.
    Evita separacion perfecta cuando hay muchos tipos con pocos registros.
    """
    df = df.copy()
    conteo = df[col].value_counts()
    raras = conteo[conteo < min_obs].index
    df.loc[df[col].isin(raras), col] = etiqueta
    return df


def _codificar_covariables(
    survival_df: pd.DataFrame,
    covariables: list[str],
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Selecciona covariables y codifica las categoricas como dummies.
    Siempre incluye 'tiempo' y 'evento'. Elimina una categoria por variable
    (drop_first=True) para evitar multicolinealidad perfecta.
    Agrupa categorias con menos de min_obs observaciones en 'Otro'.
    """
    cols = ["tiempo", "evento"] + covariables
    df = survival_df[cols].copy()

    cat_cols = df[covariables].select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df = _agrupar_categorias_raras(df, col, min_obs=min_obs)

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=float)

    return df


# -- 2b. Inspeccion del dataset de supervivencia ------------------------------

def describir_survival(survival_df: pd.DataFrame) -> None:
    """
    Imprime un resumen del dataset: totales, eventos, distribucion de tiempo
    y desglose por fuente y tipo_relleno.
    """
    n       = len(survival_df)
    eventos = survival_df["evento"].sum()
    cens    = n - eventos

    print(f"{'='*55}")
    print(f"  Dataset de supervivencia: {n} columnas")
    print(f"  Eventos (fallo SST) : {eventos}  ({eventos/n*100:.1f}%)")
    print(f"  Censuradas          : {cens}  ({cens/n*100:.1f}%)")
    print(f"{'='*55}")

    print("\n  Tiempo (inyecciones) -- fallidas vs censuradas:")
    print(
        survival_df.groupby("evento")["tiempo"]
        .describe()
        .rename(index={0: "censurada", 1: "fallida"})
        .round(1)
        .to_string()
    )

    print("\n  Por fuente:")
    resumen_fuente = (
        survival_df.groupby("fuente")
        .agg(
            n=("columna_id", "count"),
            eventos=("evento", "sum"),
            t_mediana=("tiempo", "median"),
            t_max=("tiempo", "max"),
        )
        .rename(columns={"t_mediana": "mediana_inj", "t_max": "max_inj"})
    )
    resumen_fuente["tasa_fallo"] = (
        resumen_fuente["eventos"] / resumen_fuente["n"] * 100
    ).round(1).astype(str) + "%"
    print(resumen_fuente.to_string())

    if "tipo_relleno" in survival_df.columns:
        print("\n  Tipos de relleno con al menos 1 evento:")
        resumen_tipo = (
            survival_df.groupby("tipo_relleno")
            .agg(
                n=("columna_id", "count"),
                eventos=("evento", "sum"),
                mediana_inj=("tiempo", "median"),
            )
            .query("eventos > 0")
            .sort_values("eventos", ascending=False)
        )
        resumen_tipo["tasa_fallo"] = (
            resumen_tipo["eventos"] / resumen_tipo["n"] * 100
        ).round(1).astype(str) + "%"
        print(resumen_tipo.to_string())
    print(f"{'='*55}")


# -- 3. Kaplan-Meier ----------------------------------------------------------

def kaplan_meier(
    survival_df: pd.DataFrame,
    grupo: str | None = None,
) -> dict[str, KaplanMeierFitter]:
    """
    Ajusta curvas de Kaplan-Meier, opcionalmente estratificadas por una variable.

    Parameters
    ----------
    survival_df : output de preparar_survival()
    grupo       : columna para estratificar (ej. "fuente", "tipo_relleno").
                  Si None, ajusta una unica curva global.

    Returns
    -------
    dict con un KaplanMeierFitter por cada valor del grupo (o "global").
    """
    if grupo is None:
        kmf = KaplanMeierFitter()
        kmf.fit(survival_df["tiempo"], event_observed=survival_df["evento"], label="Global")
        return {"Global": kmf}

    fitters = {}
    for valor, grupo_df in survival_df.groupby(grupo):
        kmf = KaplanMeierFitter()
        kmf.fit(grupo_df["tiempo"], event_observed=grupo_df["evento"], label=str(valor))
        fitters[str(valor)] = kmf
    return fitters


def plot_survival_data(survival_df: pd.DataFrame) -> None:
    """
    Visualiza el dataset de supervivencia crudo antes del modelado.

    Subplots
    --------
    1. Strip plot: tiempo por fuente, censuradas (gris) vs eventos (rojo)
    2. Histograma: distribucion de tiempo para censuradas vs fallidas
    3. Barras: eventos y censuradas por tipo_relleno (solo los con >= 1 evento)
    """
    PLOTS_DIR = Path(__file__).parent.parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    COLOR_EVENTO    = "#C62828"
    COLOR_CENSURADA = "#90A4AE"

    df = survival_df.copy()
    df["estado"] = df["evento"].map({1: "Fallo SST", 0: "Censurada"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # -- 1. Strip plot por fuente ---------------------------------------------
    ax = axes[0]
    fuentes = sorted(df["fuente"].unique())
    y_pos   = {f: i for i, f in enumerate(fuentes)}
    cens = df[df["evento"] == 0]
    evts = df[df["evento"] == 1]
    jitter = lambda n: np.random.default_rng(42).uniform(-0.25, 0.25, n)

    ax.scatter(
        cens["tiempo"],
        [y_pos[f] for f in cens["fuente"]] + jitter(len(cens)),
        color=COLOR_CENSURADA, marker="|", s=60, linewidths=1.2,
        alpha=0.6, label="Censurada",
    )
    ax.scatter(
        evts["tiempo"],
        [y_pos[f] for f in evts["fuente"]] + jitter(len(evts)),
        color=COLOR_EVENTO, marker="x", s=60, linewidths=1.5,
        alpha=0.9, label="Fallo SST",
    )
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Inyecciones acumuladas")
    ax.set_title("Observaciones por fuente")
    ax.legend(loc="upper right", fontsize=9)

    # -- 2. Histograma censuradas vs fallidas ---------------------------------
    ax = axes[1]
    ax.hist(cens["tiempo"], bins=30, color=COLOR_CENSURADA, alpha=0.7,
            label=f"Censurada (n={len(cens)})", density=True)
    ax.hist(evts["tiempo"], bins=30, color=COLOR_EVENTO, alpha=0.8,
            label=f"Fallo SST (n={len(evts)})", density=True)
    for t in evts["tiempo"]:
        ax.axvline(t, color=COLOR_EVENTO, alpha=0.25, linewidth=0.7, ymin=0, ymax=0.06)
    ax.set_xlabel("Inyecciones acumuladas")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribucion del tiempo observado")
    ax.legend(fontsize=9)

    # -- 3. Barras por tipo_relleno (solo los con >= 1 evento) ----------------
    ax = axes[2]
    col_grupo = "tipo_relleno" if "tipo_relleno" in df.columns else "tipo_columna"
    tipos_con_evento = (
        df.groupby(col_grupo)["evento"].sum()
        .reset_index()
        .query("evento > 0")
        .sort_values("evento", ascending=True)
    )
    conteo = (
        df[df[col_grupo].isin(tipos_con_evento[col_grupo])]
        .groupby([col_grupo, "estado"])
        .size()
        .unstack(fill_value=0)
        .reindex(tipos_con_evento[col_grupo])
    )
    conteo.plot.barh(
        stacked=True,
        color={"Censurada": COLOR_CENSURADA, "Fallo SST": COLOR_EVENTO},
        ax=ax, legend=True,
    )
    ax.set_xlabel("Cantidad de columnas")
    ax.set_ylabel("")
    ax.set_title(f"Eventos por {col_grupo}\n(con >= 1 fallo SST)")
    ax.legend(fontsize=9)

    fig.suptitle("Dataset de supervivencia -- vista previa al modelado", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_data.jpg", dpi=150, bbox_inches="tight")
    plt.show()


def plot_kaplan_meier(survival_df: pd.DataFrame) -> None:
    """
    Genera una figura con tres subplots de Kaplan-Meier:
      1. Curva global
      2. Estratificado por fuente (con test log-rank)
      3. Estratificado por tipo_relleno (tipos con >= 3 eventos)
    """
    PLOTS_DIR = Path(__file__).parent.parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # -- 1. Curva global ------------------------------------------------------
    ax = axes[0]
    kmfs = kaplan_meier(survival_df)
    kmfs["Global"].plot_survival_function(ax=ax, ci_show=True, color="#1565C0")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Supervivencia global")
    ax.set_xlabel("Inyecciones acumuladas")
    ax.set_ylabel("Probabilidad de supervivencia")
    ax.set_ylim(0, 1.05)
    n = len(survival_df)
    e = survival_df["evento"].sum()
    ax.text(0.97, 0.97, f"n={n}  eventos={e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9)

    # -- 2. Por fuente --------------------------------------------------------
    ax = axes[1]
    paleta_fuente = {"Fuente A": "#1565C0", "Fuente B": "#2E7D32", "Fuente C": "#C62828"}
    kmfs_fuente = kaplan_meier(survival_df, grupo="fuente")
    for label, kmf in kmfs_fuente.items():
        kmf.plot_survival_function(ax=ax, ci_show=True, color=paleta_fuente.get(label))

    fuentes_con_eventos = survival_df.groupby("fuente")["evento"].sum()
    fuentes_con_eventos = fuentes_con_eventos[fuentes_con_eventos > 0].index.tolist()
    if len(fuentes_con_eventos) >= 2:
        sub = survival_df[survival_df["fuente"].isin(fuentes_con_eventos)]
        lr = multivariate_logrank_test(sub["tiempo"], sub["fuente"], sub["evento"])
        ax.text(0.97, 0.97, f"log-rank p={lr.p_value:.3f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title("Supervivencia por fuente")
    ax.set_xlabel("Inyecciones acumuladas")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.05)

    # -- 3. Por tipo_relleno --------------------------------------------------
    ax = axes[2]
    col_grupo = "tipo_relleno" if "tipo_relleno" in survival_df.columns else "tipo_columna"
    eventos_por_tipo = survival_df.groupby(col_grupo)["evento"].sum()
    tipos_relevantes = eventos_por_tipo[eventos_por_tipo >= 3].index.tolist()
    sub_tipo = survival_df.copy()
    sub_tipo["grupo_tipo"] = sub_tipo[col_grupo].where(
        sub_tipo[col_grupo].isin(tipos_relevantes), "Resto"
    )
    kmfs_tipo = kaplan_meier(sub_tipo, grupo="grupo_tipo")

    paleta_relleno = {
        "C18":           "#1565C0",
        "IonExchange":   "#C62828",
        "SizeExclusion": "#2E7D32",
        "Silica":        "#F57F17",
        "C8":            "#6A1B9A",
        "Resto":         "#90A4AE",
    }
    for label, kmf in sorted(kmfs_tipo.items()):
        color = paleta_relleno.get(label, "#78909C")
        kmf.plot_survival_function(ax=ax, ci_show=(label != "Resto"), color=color)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(f"Supervivencia por {col_grupo}\n(tipos con >= 3 eventos)")
    ax.set_xlabel("Inyecciones acumuladas")
    ax.set_ylabel("")
    ax.set_ylim(0, 1.05)

    for ax in axes:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Analisis de supervivencia -- columnas HPLC", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "survival_kaplan_meier.jpg", dpi=150, bbox_inches="tight")
    plt.show()


# -- 4. Cox Proportional Hazards ----------------------------------------------

def cox_ph(
    survival_df: pd.DataFrame,
    covariables: list[str],
    penalizer: float = 0.1,
    min_obs: int = 5,
    summary: bool = False,
) -> CoxPHFitter:
    """
    Ajusta un modelo de Cox con las covariables indicadas.

    Parameters
    ----------
    survival_df  : output de preparar_survival()
    covariables  : lista de columnas a incluir (ej. ["fuente", "tipo_relleno"]).
    penalizer    : regularizacion L2 (default 0.1).
    min_obs      : categorias con menos observaciones se agrupan en "Otro".
    summary      : si True, imprime un resumen del modelo ajustado.

    Returns
    -------
    CoxPHFitter ajustado (lifelines).
    """
    df = _codificar_covariables(survival_df, covariables, min_obs=min_obs)

    modelo = CoxPHFitter(penalizer=penalizer)
    modelo.fit(df, duration_col="tiempo", event_col="evento")

    if summary:
        n, e = len(df), int(df["evento"].sum())
        print(f"\n{'='*60}")
        print(f"  Cox Proportional Hazards")
        print(f"  Covariables : {covariables}")
        print(f"  n={n}  eventos={e}  penalizer={penalizer}")
        print(f"  C-index     : {modelo.concordance_index_:.4f}")
        print(f"  AIC         : {modelo.AIC_:.2f}")
        print(f"  Log-lik     : {modelo.log_likelihood_:.2f}")
        print(f"{'='*60}")
        modelo.print_summary()

    return modelo


# -- 5. Modelos AFT -----------------------------------------------------------

def weibull_aft(
    survival_df: pd.DataFrame,
    covariables: list[str],
    penalizer: float = 0.1,
    min_obs: int = 5,
    summary: bool = False,
    ancillary: bool = False,
) -> WeibullAFTFitter:
    """
    Ajusta un modelo parametrico Weibull AFT.

    Parameters
    ----------
    survival_df  : output de preparar_survival()
    covariables  : lista de columnas a incluir como predictores.
    penalizer    : regularizacion L2 (default 0.1).
    min_obs      : categorias con menos observaciones se agrupan en "Otro".
    summary      : si True, imprime un resumen del modelo ajustado.
    ancillary    : si True, modela el parametro de forma (rho) en funcion de
                   las mismas covariables.

    Returns
    -------
    WeibullAFTFitter ajustado (lifelines).
    """
    df = _codificar_covariables(survival_df, covariables, min_obs=min_obs)

    modelo = WeibullAFTFitter(penalizer=penalizer)
    modelo.fit(df, duration_col="tiempo", event_col="evento", ancillary=ancillary)

    if summary:
        n, e = len(df), int(df["evento"].sum())
        anc_str = " + ancillary" if ancillary else ""
        print(f"\n{'='*60}")
        print(f"  Weibull AFT{anc_str}")
        print(f"  Covariables : {covariables}")
        print(f"  n={n}  eventos={e}  penalizer={penalizer}")
        print(f"  C-index     : {modelo.concordance_index_:.4f}")
        print(f"  AIC         : {modelo.AIC_:.2f}")
        print(f"  Log-lik     : {modelo.log_likelihood_:.2f}")
        print(f"{'='*60}")
        modelo.print_summary()

    return modelo


def lognormal_aft(
    survival_df: pd.DataFrame,
    covariables: list[str],
    penalizer: float = 0.1,
    min_obs: int = 5,
    summary: bool = False,
    ancillary: bool = False,
) -> LogNormalAFTFitter:
    """
    Ajusta un modelo parametrico Log-Normal AFT. Modelo principal de prediccion.

    El hazard log-normal es no monotono (sube y luego baja), consistente con
    el patron de fallo temprano observado en los datos. Seleccionado frente al
    Weibull AFT por AIC y C-index CV, y frente al Mixture Cure Model por ser
    identificable con censuramiento administrativo (8.6% eventos).

    Parameters
    ----------
    survival_df  : output de preparar_survival()
    covariables  : lista de columnas a incluir como predictores.
    penalizer    : regularizacion L2 (default 0.1).
    min_obs      : categorias con menos observaciones se agrupan en "Otro".
    summary      : si True, imprime un resumen del modelo ajustado.
    ancillary    : si True, modela sigma en funcion de las covariables.
                   Aborda heteroscedasticidad entre grupos.

    Returns
    -------
    LogNormalAFTFitter ajustado (lifelines). Estimacion por MLE.
    """
    df = _codificar_covariables(survival_df, covariables, min_obs=min_obs)

    modelo = LogNormalAFTFitter(penalizer=penalizer)
    modelo.fit(df, duration_col="tiempo", event_col="evento", ancillary=ancillary)

    if summary:
        n, e = len(df), int(df["evento"].sum())
        anc_str = " + ancillary" if ancillary else ""
        print(f"\n{'='*60}")
        print(f"  Log-Normal AFT{anc_str}")
        print(f"  Covariables : {covariables}")
        print(f"  n={n}  eventos={e}  penalizer={penalizer}")
        print(f"  C-index     : {modelo.concordance_index_:.4f}")
        print(f"  AIC         : {modelo.AIC_:.2f}")
        print(f"  Log-lik     : {modelo.log_likelihood_:.2f}")
        print(f"{'='*60}")
        modelo.print_summary()

    return modelo


def generalized_gamma_aft(
    survival_df: pd.DataFrame,
    covariables: list[str] | None = None,
    summary: bool = False,
) -> GeneralizedGammaFitter:
    """
    Ajusta un Generalized Gamma univariado para seleccionar distribucion.
    La familia generalizada anida Weibull, Log-Normal y Gamma.

    Interpretacion del parametro lambda_:
      lambda_ ~  0  -> Log-Normal favorecida
      lambda_ ~  1  -> Weibull favorecida
      lambda_ ~ -1  -> Weibull invertida

    Si el IC 95% de lambda_ contiene 0 (1), Log-Normal (Weibull) no se
    distingue estadisticamente del modelo general. Si excluye ambos, ninguna
    subfamilia es apropiada (considerar log-logistico).

    Parameters
    ----------
    survival_df : output de preparar_survival()
    covariables : ignorado (se mantiene por consistencia de firma).
    summary     : si True, imprime resumen y veredicto de subfamilia.

    Returns
    -------
    GeneralizedGammaFitter ajustado (lifelines).
    """
    _ = covariables
    modelo = GeneralizedGammaFitter()
    modelo.fit(survival_df["tiempo"], event_observed=survival_df["evento"])

    if summary:
        n, e = len(survival_df), int(survival_df["evento"].sum())
        print(f"\n{'='*60}")
        print(f"  Generalized Gamma (univariado)  |  diagnostico de distribucion")
        print(f"  n={n}  eventos={e}")
        print(f"  AIC         : {modelo.AIC_:.2f}")
        print(f"  Log-lik     : {modelo.log_likelihood_:.2f}")
        print(f"{'='*60}")
        modelo.print_summary()

        try:
            lam_hat = float(modelo.lambda_)
            lam_se  = float(modelo.summary.loc["lambda_", "se(coef)"])
            ci_low  = lam_hat - 1.96 * lam_se
            ci_high = lam_hat + 1.96 * lam_se

            print(f"\n  Diagnostico de subfamilia (parametro lambda_):")
            print(f"    estimacion : {lam_hat:+.3f}")
            print(f"    IC 95%     : [{ci_low:+.3f}, {ci_high:+.3f}]")

            contiene_0 = ci_low <= 0 <= ci_high
            contiene_1 = ci_low <= 1 <= ci_high

            if contiene_0 and not contiene_1:
                veredicto = "Log-Normal favorecida (IC contiene 0, excluye 1)"
            elif contiene_1 and not contiene_0:
                veredicto = "Weibull favorecida (IC contiene 1, excluye 0)"
            elif contiene_0 and contiene_1:
                veredicto = "Datos insuficientes para distinguir Log-Normal vs Weibull"
            else:
                veredicto = "Ni Log-Normal ni Weibull son apropiadas -- considerar log-logistico"
            print(f"    Veredicto  : {veredicto}")
            print(f"{'='*60}")
        except (KeyError, AttributeError) as err:
            print(f"  No se pudo extraer lambda_: {err}")
            print(f"{'='*60}")

    return modelo


def comparar_modelos(
    survival_df: pd.DataFrame,
    covariables: list[str],
    k: int = 5,
    penalizer: float = 0.1,
    min_obs: int = 5,
) -> pd.DataFrame:
    """
    Ajusta Weibull AFT y Log-Normal AFT y compara su ajuste.

    Metricas reportadas
    -------------------
    log_likelihood  : log-verosimilitud (MLE, in-sample) -- mayor es mejor
    aic             : criterio de informacion de Akaike (in-sample) -- menor es mejor
    c_index_insample: concordance index in-sample -- solo referencia
    c_index_cv      : C-index medio OOS via k-fold CV -- metrica principal

    Returns
    -------
    DataFrame con una fila por modelo y las cuatro metricas.
    """
    wb = weibull_aft(survival_df, covariables, penalizer=penalizer, min_obs=min_obs)
    ln = lognormal_aft(survival_df, covariables, penalizer=penalizer, min_obs=min_obs)

    eval_wb = evaluar_modelo(wb, survival_df, covariables, k=k, penalizer=penalizer, min_obs=min_obs)
    eval_ln = evaluar_modelo(ln, survival_df, covariables, k=k, penalizer=penalizer, min_obs=min_obs)

    resumen = pd.DataFrame(
        {
            "log_likelihood":   [eval_wb["log_likelihood"],   eval_ln["log_likelihood"]],
            "aic":              [eval_wb["aic"],               eval_ln["aic"]],
            "c_index_insample": [eval_wb["c_index_insample"], eval_ln["c_index_insample"]],
            "c_index_cv":       [eval_wb["c_index_cv_mean"],  eval_ln["c_index_cv_mean"]],
            "c_index_cv_std":   [eval_wb["c_index_cv_std"],   eval_ln["c_index_cv_std"]],
        },
        index=["Weibull AFT", "Log-Normal AFT"],
    )

    mejor_aic = resumen["aic"].idxmin()
    mejor_cv  = resumen["c_index_cv"].idxmax()
    print(f"\n{'='*60}")
    print(f"  Comparacion de modelos AFT  |  covariables: {covariables}")
    print(f"{'='*60}")
    print(resumen.round(4).to_string())
    print(f"\n  Mejor AIC        : {mejor_aic}")
    print(f"  Mejor C-index CV : {mejor_cv}")
    print(f"{'='*60}")

    return resumen


# -- 6. Verificacion del supuesto AFT -----------------------------------------

def verificar_supuesto_aft(
    modelo: LogNormalAFTFitter | WeibullAFTFitter,
    survival_df: pd.DataFrame,
    covariables: list[str],
    min_obs: int = 5,
) -> None:
    """
    Diagnostico grafico del supuesto AFT con tres subplots.

    Subplots
    --------
    1. Probability plot log-normal por grupo
       x = log(t),  y = Phi^-1(1 - S_KM(t))
       Supuesto OK si las lineas son rectas Y paralelas entre grupos.

    2. Log-log plot por grupo (log(t) vs log(-log(S_KM(t))))
       Lineas paralelas en log-log -> supuesto PH.
       Lineas paralelas en probability plot -> supuesto AFT.

    3. Residuos Cox-Snell
       r_i = -log(S_hat(t_i | x_i)).  Si el modelo ajusta, r_i ~ Exp(1).
       KM(r) vs -log(KM(r)) debe seguir la diagonal y = x.

    Parameters
    ----------
    modelo      : modelo AFT ajustado
    survival_df : output de preparar_survival()
    covariables : mismas covariables usadas al ajustar el modelo
    min_obs     : mismo valor usado al ajustar el modelo
    """
    PLOTS_DIR = Path(__file__).parent.parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)

    df_enc    = _codificar_covariables(survival_df, covariables, min_obs=min_obs)
    colores   = plt.cm.tab10.colors
    grupo_var = "fuente" if "fuente" in covariables else covariables[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def _km_series(sub: pd.DataFrame):
        kmf = KaplanMeierFitter()
        kmf.fit(sub["tiempo"], event_observed=sub["evento"])
        sf   = kmf.survival_function_.squeeze()
        t    = sf.index.values.astype(float)
        s    = sf.values.astype(float)
        mask = (t > 0) & (s > 0.02) & (s < 0.98)
        return t[mask], s[mask]

    grupos = sorted(survival_df[grupo_var].unique())

    # -- 1. Probability plot log-normal ---------------------------------------
    ax = axes[0]
    for i, g in enumerate(grupos):
        sub = survival_df[survival_df[grupo_var] == g]
        if sub["evento"].sum() == 0:
            continue
        t, s = _km_series(sub)
        if len(t) < 3:
            continue
        x = np.log(t)
        y = scipy_norm.ppf(1 - s)
        coef   = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 80)
        ax.scatter(x, y, color=colores[i % 10], s=18, alpha=0.7, label=str(g))
        ax.plot(x_line, np.polyval(coef, x_line),
                color=colores[i % 10], linewidth=1.2, linestyle="--")

    ax.set_xlabel("log(inyecciones)")
    ax.set_ylabel("Phi^-1(1 - S(t))")
    ax.set_title(f"Probability plot log-normal\npor {grupo_var}")
    ax.legend(fontsize=8)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")

    # -- 2. Log-log plot ------------------------------------------------------
    ax = axes[1]
    for i, g in enumerate(grupos):
        sub = survival_df[survival_df[grupo_var] == g]
        if sub["evento"].sum() == 0:
            continue
        t, s = _km_series(sub)
        if len(t) < 3:
            continue
        ax.plot(np.log(t), np.log(-np.log(s)),
                color=colores[i % 10], linewidth=1.4, label=str(g))

    ax.set_xlabel("log(inyecciones)")
    ax.set_ylabel("log(-log(S(t)))")
    ax.set_title(f"Log-log plot\npor {grupo_var}")
    ax.legend(fontsize=8)

    # -- 3. Residuos Cox-Snell ------------------------------------------------
    ax = axes[2]
    try:
        cs_res   = modelo.compute_residuals(df_enc, kind="cox_snell")
        residuos = cs_res.squeeze().values
    except Exception:
        sf_pred  = modelo.predict_survival_function(df_enc)
        tiempos  = df_enc["tiempo"].values
        residuos = np.array([
            float(np.interp(
                tiempos[i], sf_pred.index,
                -np.log(sf_pred.iloc[:, i].clip(lower=1e-9))
            ))
            for i in range(len(df_enc))
        ])

    kmf_cs = KaplanMeierFitter()
    kmf_cs.fit(residuos, event_observed=df_enc["evento"])
    sf_cs = kmf_cs.survival_function_.squeeze()
    t_cs  = sf_cs.index.values
    s_cs  = sf_cs.values.clip(1e-9)

    ax.plot(t_cs, -np.log(s_cs), color="#1565C0", linewidth=1.5, label="KM residuos")
    lim = max(t_cs.max(), (-np.log(s_cs)).max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Exp(1) teorico")
    ax.set_xlabel("Residuo Cox-Snell r")
    ax.set_ylabel("-log(KM(r))")
    ax.set_title("Residuos Cox-Snell\n(diagonal = ajuste perfecto)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    nombre = type(modelo).__name__.replace("Fitter", "")
    fig.suptitle(f"Verificacion supuesto AFT -- {nombre}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "aft_supuesto.jpg", dpi=150, bbox_inches="tight")
    plt.show()


# -- 7. Evaluacion ------------------------------------------------------------

def _cv_scores(
    ModelClass: type,
    df_enc: pd.DataFrame,
    penalizer: float,
    k: int,
) -> list[float]:
    """
    Corre k-fold CV de lifelines y devuelve los C-index por fold.

    k_fold_cross_validation asume convencion Cox (prediccion alta = mayor riesgo).
    Los AFT predicen tiempo (prediccion alta = menor riesgo), por lo que el signo
    queda invertido. Se niega para recuperar el C-index real.
    """
    from lifelines.utils import k_fold_cross_validation
    raw = k_fold_cross_validation(
        ModelClass(penalizer=penalizer),
        df_enc,
        duration_col="tiempo",
        event_col="evento",
        k=k,
    )
    return [-s for s in raw]


def buscar_penalizer(
    survival_df: pd.DataFrame,
    covariables: list[str],
    ModelClass: type,
    penalizers: list[float] | None = None,
    k: int = 5,
    min_obs: int = 5,
) -> tuple[float, pd.DataFrame]:
    """
    Busca el penalizer que maximiza el C-index OOS via k-fold CV.

    Parameters
    ----------
    survival_df : output de preparar_survival()
    covariables : covariables a incluir
    ModelClass  : CoxPHFitter, WeibullAFTFitter o LogNormalAFTFitter
    penalizers  : valores a evaluar. Por defecto grilla log-uniforme en [1e-3, 10].
    k           : numero de folds para CV
    min_obs     : mismo valor que se usara al ajustar el modelo final

    Returns
    -------
    best_penalizer : float con el valor optimo
    resultados     : DataFrame con (penalizer, c_index_mean, c_index_std) ordenado

    Example
    -------
    best, tabla = buscar_penalizer(survival, ["fuente", "tipo_relleno"], LogNormalAFTFitter)
    modelo = lognormal_aft(survival, ["fuente", "tipo_relleno"], penalizer=best)
    """
    if penalizers is None:
        penalizers = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    df_enc = _codificar_covariables(survival_df, covariables, min_obs=min_obs)

    filas = []
    for p in penalizers:
        s = pd.Series(_cv_scores(ModelClass, df_enc, penalizer=p, k=k))
        filas.append({"penalizer": p, "c_index_mean": s.mean(), "c_index_std": s.std()})

    resultados = (
        pd.DataFrame(filas)
        .sort_values("c_index_mean", ascending=False)
        .reset_index(drop=True)
    )
    best = float(resultados.loc[0, "penalizer"])

    nombre = ModelClass.__name__.replace("Fitter", "")
    print(f"\n{'='*60}")
    print(f"  Busqueda de penalizer -- {nombre}  |  CV {k}-fold")
    print(f"{'='*60}")
    print(resultados.round(4).to_string(index=False))
    print(f"\n  Penalizer optimo: {best}")
    print(f"{'='*60}")

    return best, resultados


def evaluar_modelo(
    modelo: object,
    survival_df: pd.DataFrame,
    covariables: list[str],
    k: int = 5,
    penalizer: float = 0.1,
    min_obs: int = 5,
) -> dict:
    """
    Calcula metricas de ajuste con C-index out-of-sample via k-fold CV.

    Parameters
    ----------
    modelo      : modelo ajustado (WeibullAFTFitter o LogNormalAFTFitter).
    survival_df : output de preparar_survival()
    covariables : mismas covariables usadas al ajustar el modelo
    k           : numero de folds (default 5)
    penalizer   : mismo valor usado al ajustar el modelo original
    min_obs     : mismo valor usado al ajustar el modelo original

    Returns
    -------
    dict con:
      c_index_insample -- C-index sobre el dataset completo (optimista, solo referencia)
      c_index_cv_mean  -- C-index medio OOS (metrica principal a reportar)
      c_index_cv_std   -- desviacion estandar entre folds
      c_index_cv_folds -- C-index por fold
      aic              -- AIC in-sample (valido para comparar modelos)
      log_likelihood   -- log-verosimilitud in-sample
    """
    ModelClass = type(modelo)
    df_enc     = _codificar_covariables(survival_df, covariables, min_obs=min_obs)
    scores     = _cv_scores(ModelClass, df_enc, penalizer=penalizer, k=k)

    scores_s = pd.Series(scores)
    result = {
        "c_index_insample": modelo.concordance_index_,
        "c_index_cv_mean":  float(scores_s.mean()),
        "c_index_cv_std":   float(scores_s.std()),
        "c_index_cv_folds": list(scores),
        "aic":              modelo.AIC_,
        "log_likelihood":   modelo.log_likelihood_,
    }

    nombre = ModelClass.__name__.replace("Fitter", "")
    print(f"\n{'='*60}")
    print(f"  {nombre}  |  CV {k}-fold (lifelines)")
    print(f"  C-index in-sample : {result['c_index_insample']:.4f}  (optimista)")
    print(f"  C-index CV media  : {result['c_index_cv_mean']:.4f}  +/- {result['c_index_cv_std']:.4f}")
    print(f"  Por fold          : {[round(c, 3) for c in scores]}")
    print(f"  AIC               : {result['aic']:.2f}  (in-sample, valido para comparar)")
    print(f"  Log-likelihood    : {result['log_likelihood']:.2f}")
    print(f"{'='*60}")

    return result


# -- 8. Prediccion ------------------------------------------------------------

def predecir_vida_util(
    modelo: WeibullAFTFitter | LogNormalAFTFitter,
    columna: pd.DataFrame | str,
    percentiles: list[float] = [0.25, 0.50, 0.75],
    survival_df: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Devuelve los percentiles de vida util esperada en inyecciones.

    Parameters
    ----------
    modelo      : WeibullAFTFitter o LogNormalAFTFitter ajustado
    columna     : DataFrame de una fila con covariables crudas (fuente, tipo_relleno, ...)
                  O un columna_id como string (ej. "C218-1"). En ese caso se requiere
                  survival_df para hacer el lookup de atributos.
    percentiles : probabilidades de supervivencia a invertir.
                  p=0.25 -> estimacion optimista (75% ha fallado).
                  p=0.50 -> mediana (50% ha fallado).
                  p=0.75 -> umbral conservador (solo 25% ha fallado, alerta temprana).
    survival_df : output de preparar_survival(). Requerido cuando columna es un string.

    Returns
    -------
    pd.Series con indice = percentiles y valores = inyecciones estimadas.

    Notes
    -----
    - Categorias no vistas durante el entrenamiento se tratan como "Otro".
    - Si la categoria era la de referencia (drop_first), todos sus dummies = 0.
    """
    if isinstance(columna, str):
        if survival_df is None:
            raise ValueError("survival_df es requerido cuando columna es un columna_id.")
        fila = survival_df[survival_df["columna_id"] == columna]
        if fila.empty:
            raise ValueError(f"columna_id '{columna}' no encontrado en survival_df.")
        idx_modelo = modelo.params_.index
        if isinstance(idx_modelo, pd.MultiIndex):
            covs_modelo = [c for _, c in idx_modelo if c != "Intercept"]
        else:
            covs_modelo = [c for c in idx_modelo if c != "Intercept"]
        covs_raw = list(dict.fromkeys(c.rsplit("_", 1)[0] for c in covs_modelo))
        cols_disponibles = [c for c in covs_raw if c in fila.columns]
        columna = fila[cols_disponibles].iloc[[0]].reset_index(drop=True)

    idx = modelo.params_.index
    if isinstance(idx, pd.MultiIndex):
        all_covs = [col for _, col in idx if col != "Intercept"]
    else:
        all_covs = [col for col in idx if col != "Intercept"]

    seen, feature_cols = set(), []
    for c in all_covs:
        if c not in seen:
            seen.add(c)
            feature_cols.append(c)

    row = pd.DataFrame(0.0, index=[0], columns=feature_cols)

    for raw_col in columna.columns:
        val   = str(columna[raw_col].iloc[0])
        dummy = f"{raw_col}_{val}"
        if dummy in row.columns:
            row[dummy] = 1.0
        else:
            otro = f"{raw_col}_Otro"
            if otro in row.columns:
                row[otro] = 1.0

    resultados = {
        p: float(modelo.predict_percentile(row, p=p).iloc[0])
        for p in percentiles
    }
    return pd.Series(resultados, name="inyecciones_estimadas")


