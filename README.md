# **IBEX Banks RNN Forecast - Predicción de series temporales financieras para BBVA y Banco Santander**

El presente proyecto aborda la **predicción del precio de cierre de las acciones de BBVA (BBVA.MC)** y **Banco Santander (SAN.MC)** mediante el uso de **modelos de aprendizaje profundo basados en redes neuronales recurrentes (RNN, LSTM, GRU)**.

El objetivo consiste en **anticipar la evolución de los precios bursátiles a corto plazo (3–5 días)** integrando tanto información **endógena** (histórica del propio activo) como **exógena** (indicadores macroeconómicos y eventos relevantes).

La iniciativa se enmarca dentro de un caso de uso académico orientado a demostrar la viabilidad técnica de sistemas de predicción financiera basados en series temporales, aplicando criterios de reproducibilidad, trazabilidad y transparencia de datos.

## **Contexto y motivación**

El comportamiento de los activos bancarios españoles está fuertemente condicionado por la política monetaria del **Banco Central Europeo (BCE)**, los ciclos económicos europeos, la inflación y los eventos financieros globales.
Modelar estos factores de manera conjunta permite evaluar **escenarios futuros**, mejorar la toma de decisiones financieras y explorar la **capacidad predictiva de arquitecturas recurrentes** frente a métodos estadísticos tradicionales.

El caso de estudio se centra en:

* **BBVA**, representando una entidad con alta exposición internacional y fuerte sensibilidad a los tipos de interés.
* **Banco Santander**, cuyo volumen y diversificación geográfica exigen modelar dinámicas más estables y amortiguadas.

## **Objetivos**

**Objetivo general:** Desarrollar y validar un sistema reproducible de **predicción de precios financieros** mediante redes neuronales recurrentes, aplicable a entidades del IBEX 35.

**Objetivos específicos:**

1. Recolectar, validar y unificar datos financieros históricos y variables macroeconómicas.
2. Generar variables exógenas que representen eventos económicos relevantes (crisis, políticas del BCE, elecciones).
3. Entrenar y comparar modelos RNN, LSTM y GRU para BBVA y Santander.
4. Evaluar el rendimiento predictivo mediante métricas cuantitativas (RMSE, MAE, R²).
5. Incorporar un módulo de simulación de escenarios y una aplicación Flask para consulta interactiva.
6. Asegurar trazabilidad completa del proceso mediante registros de linaje de datos.

## **Estructura del repositorio**

```powershell
ibex-banks-rnn/
├─ .cache/                      # Capas intermedias no versionadas
│   ├─ raw/                     # Datos OHLCV descargados de Yahoo Finance
│   ├─ exogenous/               # Variables de eventos binarios (crisis, elecciones, BCE)
│   ├─ macro/                   # Indicadores macroeconómicos (tipos, inflación, IBEX)
│   ├─ features/                # Series finales enriquecidas con lags y factores externos
│   └─ validation/              # Informes de integridad de datos
│
├─ config/
│   ├─ data.yml                 # Tickers, fechas, calendarios y columnas relevantes
│   └─ exogenous_events.yml     # Definición de eventos macroeconómicos y shocks
│
├─ docs/
│   ├─ Use_Case_Financial_Forecasting.pdf  # Documento teórico del caso de uso
│   ├─ YFinance_Explained.md               # Descripción técnica de la fuente de datos
│
├─ notebooks/
│   ├─ 01_eda_series.ipynb       # Descomposición de series y análisis exploratorio
│   ├─ 02_exogenous_design.ipynb # Diseño de variables exógenas y validación visual
│   └─ 03_compare_rnn_lstm_gru.ipynb # Entrenamiento y comparación de arquitecturas
│
├─ reports/
│   ├─ app_bundle/               # Artefactos generados (modelos, métricas, escaladores)
│   │   ├─ BBVA.MC/
│   │   │   ├─ BBVA.MC_metrics_test.json
│   │   │   └─ modelos entrenados (.pt)
│   │   └─ SAN.MC/
│   │       ├─ SAN.MC_metrics_test.json
│   │       └─ modelos entrenados (.pt)
│   ├─ figures/                  # Gráficos de evolución y validación
│
├─ src/
│   ├─ data/                     # Ingesta y validación de datos
│   │   ├─ load_raw.py
│   │   ├─ build_exogenous.py
│   │   └─ validate_integrity.py
│   ├─ eda/                      # Ejecución de EDA
│   │   └─ run_eda.py
│   ├─ utils/                    # Módulos de utilidades
│   │   ├─ config.py
│   │   ├─ io_utils.py
│   │   ├─ logging_utils.py
│   │   └─ time_utils.py
│   └─ __init__.py
│
├─ tests/                        # Pruebas de validación
├─ Makefile                      # Automatización de pipeline
└─ README.md
```

## **Metodología de trabajo**

### **Fases del pipeline**

1. **Recolección de datos:** descarga y normalización de series financieras desde Yahoo Finance.
2. **Validación e integridad:** detección de huecos, duplicados y rangos anómalos.
3. **Análisis exploratorio:** descomposición de tendencias, estacionalidades y volatilidades.
4. **Construcción de variables exógenas:** marcaje de eventos binarios y ampliación con indicadores macroeconómicos.
5. **Transformaciones de estacionariedad:** cálculo de retornos porcentuales y diferencias temporales.
6. **Generación de lags:** creación de ventanas temporales para capturar memoria histórica.
7. **Entrenamiento de modelos recurrentes:** ajuste de hiperparámetros y optimización con GPU.
8. **Evaluación:** comparación de modelos mediante RMSE, MAE y R².
9. **Predicción y simulación:** generación de pronósticos 3–5 noviembre 2025 y escenarios alternativos.
10. **Despliegue:** aplicación Flask con inferencia en línea y visualización interactiva.

### **Fuentes de datos**

* **Yahoo Finance:** series OHLCV de BBVA.MC, SAN.MC, IBEX35 y S&P500.
* **Banco Central Europeo (BCE):** tipo de depósito e inflación armonizada (HICP).
* **Eurostat / OCDE:** PIB trimestral y tasa de desempleo.
* **Eventos históricos:** crisis dot-com, financiera global 2008, Brexit, COVID-19, guerra de Ucrania.

## **Arquitectura del modelo**

Las tres arquitecturas comparten una estructura base:

* **Entrada:** ventana temporal de tamaño *n* (lags + exógenas + macro).
* **Capa recurrente:** RNN / LSTM / GRU con 64 neuronas y `dropout=0.2`.
* **Capa densa final:** `Linear(64 → 1)` para predecir el valor de cierre siguiente.
* **Función de pérdida:** Error cuadrático medio (MSE).
* **Optimizador:** Adam (LR = 1e-3).
* **Entrenamiento:** 100 épocas máx. con *early stopping* y *batch size* = 32.

### **Particularidades**

* **RNN:** secuencia simple con memoria corta; sirve como referencia base.
* **LSTM:** introduce puertas de entrada, olvido y salida; mantiene dependencias de largo plazo.
* **GRU:** simplifica la LSTM manteniendo su capacidad de memoria; requiere menos parámetros y convergencia más rápida.

## **Resultados experimentales**

### **BBVA (BBVA.MC)**

Resultados extraídos de `reports/app_bundle/BBVA.MC/BBVA.MC_metrics_test.json`:

| Modelo | RMSE       | MAE        | R²         |
| ------ | ---------- | ---------- | ---------- |
| RNN    | 0.0968     | 0.2499     | 0.9161     |
| LSTM   | 0.1429     | 0.2959     | 0.8762     |
| GRU    | **0.0816** | **0.2128** | **0.9293** |

**Análisis:**
La arquitectura **GRU** obtiene el mejor desempeño general, reduciendo el error cuadrático medio en un **15,7 % respecto a la RNN** y en un **43 % respecto a la LSTM**.
La RNN clásica mantiene buen ajuste (R² ≈ 0.91) pero muestra sobreajuste en validación. La LSTM no mejora los resultados debido a la complejidad adicional y tamaño limitado del conjunto de entrenamiento.

### **Banco Santander (SAN.MC)**

Resultados extraídos de `reports/app_bundle/SAN.MC/SAN.MC_metrics_test.json`:

| Modelo | RMSE       | MAE        | R²         |
| ------ | ---------- | ---------- | ---------- |
| RNN    | 0.0238     | 0.1194     | 0.8667     |
| LSTM   | **0.0213** | 0.1210     | **0.8805** |
| GRU    | 0.0229     | **0.1183** | 0.8717     |

**Análisis:**
En el caso de **Santander**, la **LSTM** presenta el mejor rendimiento global (R² = 0.88), seguida muy de cerca por la GRU.
Las diferencias en error absoluto (MAE) son marginales (<3 %), lo que indica un comportamiento más estable y menos dependiente del modelo, coherente con la naturaleza más amortiguada del activo.

### **Comparativa global**

| Activo    | Modelo óptimo | R²    | RMSE   | MAE   |
| --------- | ------------- | ----- | ------ | ----- |
| BBVA      | **GRU**       | 0.929 | 0.0816 | 0.213 |
| Santander | **LSTM**      | 0.880 | 0.0213 | 0.121 |

**Conclusión técnica:**

* La **GRU** se consolida como el modelo más eficiente para series con mayor volatilidad (BBVA).
* La **LSTM** es más adecuada para series más estables (Santander).
* Ambos modelos superan a la RNN en precisión y generalización, validando el uso de arquitecturas con memoria avanzada.

## **Predicción y escenarios**

El sistema genera predicciones para el **3, 4 y 5 de noviembre de 2025**, acompañadas de un intervalo de confianza del ±3 %.
Asimismo, se incluye un módulo de **simulación de escenarios hipotéticos**, que permite modificar variables exógenas (por ejemplo, “crisis 2008 = 1” o “subida BCE = 1”) y observar el impacto en las predicciones.

Este enfoque facilita un análisis de sensibilidad sobre factores externos y permite evaluar la respuesta del modelo a perturbaciones macroeconómicas.

## **Reproducibilidad**

El proyecto implementa mecanismos de reproducibilidad total:

* Configuraciones en YAML (`config/data.yml`, `config/exogenous_events.yml`).
* Control de semillas y determinismo en PyTorch.
* Datos intermedios en `.cache/` con estructura fija.
* Registro detallado de linaje de datos (`logs/data_lineage.jsonl`).
* Versionado de modelos y métricas en `reports/app_bundle/`.
* Automatización mediante **Makefile**:

```bash
make data
make exogenous
make macro
make features
make train
make evaluate
make reports
```

## **Aplicación Flask**

Se desarrolló una aplicación web basada en **Flask**, desplegada en **HuggingFace Spaces**, que permite la interacción con los modelos entrenados y la visualización de resultados.

Características principales:

* Carga dinámica de modelos GRU (BBVA) y LSTM (SAN).
* Simulación de eventos y shocks exógenos.
* Visualización de curvas reales vs. predichas.
* Exportación de resultados a CSV y PNG.

**Enlace de despliegue:**
[https://huggingface.co/spaces/mgonzalz/ibex-banks-rnn-forecast](https://huggingface.co/spaces/mgonzalz/ibex-banks-rnn-forecast)

## **Conclusiones generales**

1. Los modelos recurrentes son capaces de capturar con precisión patrones secuenciales en series financieras.
2. La inclusión de variables exógenas y macroeconómicas mejora la estabilidad de las predicciones y la interpretabilidad.
3. La arquitectura GRU resulta más eficiente en contextos de alta variabilidad (BBVA), mientras que la LSTM ofrece mayor robustez en series más regulares (Santander).
4. La metodología implementada es completamente reproducible y auditable, conforme a buenas prácticas de ingeniería de datos.
5. El uso de escenarios contrafactuales amplía la utilidad práctica del sistema como herramienta de análisis de riesgo o simulación económica.

## **Presentación final**

La presentación ejecutiva del proyecto se encuentra en:

```bash
reports/presentation/IBEX_RNN_Presentation_Final.pdf
```

Incluye la motivación, la arquitectura general, los resultados numéricos y la comparación visual de los tres modelos por entidad.

## **Licencia**

Este repositorio se distribuye bajo licencia **MIT**, que permite el uso, modificación y redistribución con atribución al autor original.
