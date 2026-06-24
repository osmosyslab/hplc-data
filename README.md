# Modelo de Vida Útil de Columnas Cromatográficas HPLC

Modelo estadístico-probabilístico para estimar cuántas inyecciones va a durar una columna cromatográfica HPLC antes de fallar el System Suitability Test (SST), integrando datos históricos de tres laboratorios farmacéuticos.

> **Aviso de privacidad**: los datos y nombres de los laboratorios participantes son confidenciales.
> Las tres fuentes de datos están anonimizadas como **Fuente A**, **Fuente B** y **Fuente C** en todo el código y la documentación.

---

## Problema

Las columnas HPLC se degradan con el uso. Hoy el criterio de reemplazo depende de la experiencia del técnico, sin base cuantificable ni documentable. Las consecuencias:

- **Reemplazar tarde**: resultados inválidos, investigaciones OOS, riesgo regulatorio (FDA, ICH, USP)
- **Reemplazar pronto**: costo innecesario de columna y tiempo de reacondicionamiento

No existe literatura publicada que resuelva este problema con modelos predictivos de datos reales.

---

## Enfoque

Pipeline completo de ciencia de datos: ETL multi-fuente → EDA → Análisis de supervivencia → Predicción de vida útil restante (RUL).

**Modelo principal**: Log-Normal AFT (Accelerated Failure Time), elegido sobre Weibull por su hazard no monotónico, consistente con el patrón de fallos tempranos observado en los datos. El modelo de cura (Mixture Cure) fue evaluado y descartado.

**Variable objetivo**: número de inyecciones acumuladas hasta el fallo del SST (dato de supervivencia con censura por la derecha).

---

## Datos

| Fuente | Columnas | Fallos SST | Censuradas |
|--------|----------|------------|------------|
| Fuente A | 1,031 | 54 | 977 |
| Fuente B | 70 | 0 | 70 |
| Fuente C | 429 | 3 | 426 |
| **Total** | **1,530** | **57 (3.7%)** | **1,473** |

**Parámetros SST monitoreados**:
- N (platos teóricos): eficiencia de la columna. Límite: N ≥ 8,000
- Factor tailing: simetría del pico. Límite: T ≤ 2.0
- Resolución: separación entre picos. Límite: R ≥ 2.0
- %RSD: repetibilidad de inyecciones. Límite: RSD ≤ 2%

**Feature central — distancia al umbral**:

```
distancia_umbral = (valor_medido - valor_umbral) / |valor_umbral|
```

Positivo = margen de seguridad; cero = en el límite; negativo = fallo.

---

## Estructura del Proyecto

```
hplc-data/
├── main.py                    # Orquestador ETL → Modelos
├── utils.py                   # Utilidad de visualización HTML
├── requirements.txt
│
├── ETL/
│   ├── extract.py             # Carga de CSVs raw por fuente
│   ├── transform.py           # Limpieza, normalización, feature engineering
│   └── load.py                # Guardado en data/proc/ y data/proc/model_data/
│
├── model/
│   ├── survival.py            # Pipeline completo de análisis de supervivencia
│   ├── modelo_c18.py          # Pipeline especializado para columnas C-18
│   └── clasificador.py        # Clasificación de columnas por tipo de relleno
│
└── scripts/
    └── QueryCriteriosParametros.sql  # Query de extracción desde LIMS
```

> **Nota**: el directorio `data/` con los CSVs raw y procesados no está incluido
> en este repositorio (datos propietarios de los laboratorios participantes).
> Ejecutar `python main.py` con los CSVs en su lugar genera `data/proc/` automáticamente.

---

## Pipeline de Modelado (`model/survival.py`)

```
1. cargar_datos()             Carga desde model_data/
2. preparar_survival()        Construye dataset: (columna_id, tiempo, evento, covariables)
3. describir_survival()       Estadísticas descriptivas y tasa de eventos
4. plot_survival_data()       Strip plot, histograma de tiempos, KM por grupo
5. plot_kaplan_meier()        Curvas KM estratificadas por fuente y tipo de relleno
6. generalized_gamma_aft()    Diagnóstico de selección de distribución (λ≈0 → Log-Normal)
7. weibull_aft()              Ajuste Weibull AFT (baseline)
   lognormal_aft()            Ajuste Log-Normal AFT (modelo principal)
   comparar_modelos()         Comparación AIC + C-index CV
8. verificar_supuesto_aft()   Residuos Cox-Snell (deben seguir Exp(1))
9. evaluar_modelo()           C-index out-of-sample (k-fold CV, k=5)
10. predecir_vida_util()      P25 / P50 / P75 con IC95% (método delta)
```

**Covariables del modelo**:
- `fuente`: condiciones operativas varían por laboratorio
- `tipo_relleno`: C18, C8, Phenyl, CN, HILIC, Chiral, Silica, etc.

**Covariables descartadas**:
- `tipo_columna`: 74 categorías → EPV = 0.16 con 12 eventos (sobreajuste)
- Variables físicas (longitud, diámetro, tamaño de partícula): sin señal estadística (p > 0.2)

---

## Cómo Ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline completo (ETL + modelado)
python main.py
```

Configurar en `main.py`:

```python
CORRER_EDA            = False   # Visualizaciones exploratorias
CORRER_SURVIVAL       = True    # Modelos de supervivencia (activo por defecto)
CORRER_CLASIFICACION  = False   # Clustering por especificaciones físicas
```

Si los datos procesados ya existen en `data/proc/`, el ETL se omite automáticamente.

---

## Salidas del Modelo

**Predicción de vida útil** para una columna dada:

```
C218-1  [Fuente A / C18]  evento=0  t_obs=4,320
  P25 (optimista)  :   8,100  IC95% [  4,200,  15,300]
  P50 (mediana)    :  12,500  IC95% [  7,100,  22,000]
  P75 (conservador):  18,900  IC95% [ 10,400,  34,200]
```

Interpretación:
- **P25**: el 25% de columnas similares fallan antes de este número de inyecciones
- **P75**: criterio conservador; el 75% de columnas similares fallan antes

---

## Stack Tecnológico

| Librería | Uso |
|----------|-----|
| `pandas` | Manipulación de datos |
| `numpy` | Cómputo numérico |
| `lifelines` | Kaplan-Meier, Cox, Weibull AFT, Log-Normal AFT |
| `scipy` | Funciones estadísticas, método delta |
| `scikit-learn` | Clustering, TF-IDF, validación cruzada |
| `matplotlib` / `seaborn` | Visualización |

---

## Referencias de Dominio

- **HPLC**: High Performance Liquid Chromatography
- **SST**: System Suitability Test (USP <621>, ICH Q2)
- **OOS**: Out of Specification
- **RUL**: Remaining Useful Life
- **AFT**: Accelerated Failure Time (modelo de supervivencia paramétrico)
- **Censura por la derecha**: columna retirada o aún activa antes de observar el fallo
