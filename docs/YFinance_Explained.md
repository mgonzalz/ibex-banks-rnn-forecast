# **Documento Explicativo: Uso de la Librería `Yahoo Finance` y su Aplicación en el Caso de Estudio**

El presente documento tiene como finalidad explicar de forma técnica y argumentada el uso de la librería `yfinance` para la obtención de datos financieros históricos, así como su aplicación práctica dentro del caso de uso. El objetivo es garantizar la trazabilidad, coherencia metodológica y justificación de las decisiones técnicas adoptadas para la construcción del dataset base que alimentará los modelos de predicción.

## **Contexto del Caso de Uso**

El sector bancario constituye uno de los pilares fundamentales del **IBEX 35**, y dentro de este destacan **BBVA** y **Banco Santander**, dos de las entidades financieras españolas con mayor capitalización bursátil, liquidez y presencia internacional.

Estas acciones son altamente representativas del comportamiento del sistema financiero español y europeo, funcionando como indicadores adelantados del estado económico y la confianza del mercado. Por su volumen de negociación y sensibilidad ante factores macroeconómicos (tipos de interés, política monetaria del BCE, riesgo geopolítico, etc.), resultan idóneas para aplicar técnicas de **análisis predictivo basado en aprendizaje automático.**

El caso de uso tiene como propósito el desarrollo de un modelo de predicción que permita **anticipar movimientos futuros en los precios de cierre** de BBVA y Santander, apoyándose en redes neuronales recurrentes (RNN, LSTM y GRU).

El estudio contempla:

* Evaluar los puntos críticos, picos y cambios de tendencia del mercado.
* Comparar el desempeño de diferentes arquitecturas de RNN.
* Realizar predicciones a corto y medio plazo.
* Facilitar herramientas que apoyen la **toma de decisiones de inversión.**

## **Obtención de Datos con la Librería `Yahoo-Finance`**

Para el desarrollo del modelo, se emplea la librería `yfinance`, la cual permite acceder de manera programática a los datos históricos de Yahoo Finance. Su principal ventaja es la facilidad para descargar precios de apertura, cierre, máximos, mínimos, volumen de transacciones y eventos corporativos (dividendos y *stock splits*) en formato estructurado de **DataFrame**.

## **Comandos principales**

El objeto base de `yfinance` permite dos métodos equivalentes para la descarga de información:

```python
Ticker.history(period="1mo", interval="1d", start=None, end=None, prepost=False,
               actions=True, auto_adjust=False, repair=False, rounding=False,
               timeout=10, raise_errors=False)
```

```python
yf.download(period="1mo", interval="1d", start=None, end=None, prepost=False,
         actions=True, auto_adjust=True, repair=False, progress=True, timeout=10)
```

Ambos métodos ofrecen control total sobre los parámetros temporales y las opciones de ajuste de precios.

### **Parámetros más relevantes**

| Parámetro        | Valor por defecto | Descripción                                                                                  |
| ---------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| **period**       | `"1mo"`           | Descarga datos del último mes si no se especifican fechas `start` o `end`.                   |
| **interval**     | `"1d"`            | Frecuencia temporal (1 día). Puede configurarse a minutos, horas, semanas o meses.           |
| **start / end**  | `None`            | Definen el rango temporal exacto de descarga.                                                |
| **prepost**      | `False`           | Incluye o excluye datos de pre-mercado y *after-hours* (solo para acciones estadounidenses). |
| **actions**      | `True`            | Añade columnas de dividendos y *splits*.                                                     |
| **auto_adjust**  | `False`           | Si se activa, ajusta automáticamente los precios históricos por dividendos y *splits*.       |
| **timeout**      | `10`              | Tiempo máximo de espera para la respuesta del servidor.                                      |
| **raise_errors** | `False`           | Controla el comportamiento ante fallos de descarga.                                          |

## **Diferencia entre “Close” y “Adjusted Close”**

Una parte esencial del análisis de series temporales financieras es la correcta interpretación del precio de cierre. `Yahoo-Finance` devuelve dos columnas diferenciadas:

| Columna       | Significado                                     | Qué refleja                                                      |
| ------------- | ----------------------------------------------- | ---------------------------------------------------------------- |
| **Close**     | Precio de cierre real del mercado               | Valor al que finalizó la sesión bursátil.                        |
| **Adj Close** | Precio ajustado por dividendos y *stock splits* | Valor teórico que mantiene la continuidad económica de la serie. |

**Ejemplo:** Si una empresa paga un dividendo de 5 € el 10 de mayo:

| Fecha  | Close | Adj Close | Dividendo |
| ------ | ----- | --------- | --------- |
| 09-may | 100 € | 95 €      | —         |
| 10-may | 95 €  | 95 €      | 5 €       |

El **Close** baja de 100 € a 95 € reflejando el pago del dividendo, pero el **Adj Close** corrige el valor anterior para mantener la coherencia temporal.

## **Eventos Corporativos: Dividendos y Stock Splits**

| Evento          | Descripción                              | Ejemplo                          | Efecto                                                |
| --------------- | ---------------------------------------- | -------------------------------- | ----------------------------------------------------- |
| **Dividendo**   | Reparto de beneficios a los accionistas. | 0,25 €/acción.                   | El precio baja en proporción al importe.              |
| **Stock Split** | División o agrupación de acciones.       | 2:1 → una acción pasa a ser dos. | El número de acciones cambia, pero no el valor total. |

stos eventos ocurren de forma puntual (dividendos, normalmente trimestrales; splits, esporádicos). Por ello, es crucial decidir si los precios se ajustarán automáticamente (`auto_adjust=True`) o se mantendrán nominales (`auto_adjust=False`).

## **Ajuste Automático de Precios**

Cuando `auto_adjust=True`, la librería recalcula las columnas **Open, High, Low y Close**, de manera que toda la serie temporal queda ajustada por dividendos y splits, eliminando saltos artificiales y permitiendo una visión continua del rendimiento real.
Si el ajuste está desactivado (`auto_adjust=False`), se mantiene la columna **Adj Close** aparte, conservando tanto el valor nominal como el ajustado.

El ajuste se realiza mediante un **factor acumulativo** calculado por Yahoo Finance, que integra los efectos de dividendos y *splits* históricos.

## **Ventajas y Desventajas del Ajustado (*Adjusted Close*)**

En el análisis de series temporales financieras, la correcta interpretación de los precios históricos es un aspecto crucial. La columna **“Adjusted Close”** refleja los precios de cierre ajustados por dividendos y *stock splits*, proporcionando una representación más fiel del rendimiento total del activo a lo largo del tiempo.
A continuación, se presentan las principales **ventajas** y **desventajas** de su uso, ilustradas con ejemplos prácticos.

### **Ventajas del Precio Ajustado**

#### **Refleja la ganancia real del inversionista**

---
**Descripción:** El precio ajustado incorpora el efecto de los dividendos (pagos directos al accionista) y de los *stock splits* (divisiones de acciones), mostrando la rentabilidad total que habría obtenido un inversor manteniendo su posición.

**Ejemplo:** Un inversor compra una acción por **100 €**. Un año después, la acción vale **110 €** y la empresa paga un dividendo de **5 €**. El precio nominal (*Close*) indicaría una ganancia de 10 €, pero el **precio ajustado** considera también el dividendo, reflejando una ganancia total de **15 €**.

**Conclusión parcial:** El *Adjusted Close* representa con mayor precisión la **ganancia económica real** obtenida por el inversor.

#### **Facilita la comparación entre empresas**

---
**Descripción:** Algunas empresas distribuyen dividendos y otras no. Si solo se consideran los precios nominales, podría parecer que las primeras ofrecen menor rendimiento. El *Adjusted Close* permite igualar las condiciones de comparación.

**Ejemplo:**

* Empresa A: sube de 100 € a 105 € y reparte 5 € en dividendos.
* Empresa B: sube de 100 € a 110 €, sin dividendos.

A simple vista, la Empresa B parece más rentable (10 € frente a 5 €).
Sin embargo, con precios ajustados, ambas muestran un rendimiento total de **10 €**.

**Conclusión parcial:** El uso de precios ajustados **evita comparaciones erróneas** entre compañías con políticas de dividendos diferentes.

#### **Ideal para el análisis de largo plazo**

---
**Descripción:** En estudios de series de 10, 15 o más años, los *splits* y dividendos distorsionan la evolución aparente del precio. El *Adjusted Close* corrige estos efectos para mantener la coherencia temporal.

**Ejemplo:** Si una acción de Apple valía 100 € hace 15 años y posteriormente realizó divisiones 7:1 y 4:1, los precios actuales no son comparables sin ajuste. El precio ajustado **homogeneiza** toda la serie y evita falsas interpretaciones de crecimiento exponencial.

**Conclusión parcial:** El ajuste garantiza **consistencia histórica**, imprescindible para modelos predictivos basados en aprendizaje automático.

#### **Evita interpretaciones erróneas tras *stock splits***

---
**Descripción:** Cuando una empresa divide sus acciones (por ejemplo, 2 por 1), el precio se reduce a la mitad, pero el número de acciones se duplica. No hay ganancia ni pérdida real. El *Adjusted Close* reescala los precios previos al evento para mantener la continuidad del gráfico.

**Ejemplo:**

* Antes del *split*: 1 acción = 200 €
* Después del *split 2:1*: 2 acciones = 100 € cada una.
El precio ajustado **corrige** los valores anteriores para que no aparezca una caída ficticia de 200 € → 100 €.

**Conclusión parcial:**
Permite que las gráficas históricas **reflejen la continuidad económica real** de la acción sin rupturas artificiales.

### **Desventajas del Precio Ajustado**

#### **Borra la historia real del mercado**

---
**Descripción:** El ajuste modifica los precios antiguos, eliminando los valores reales a los que se negoció la acción en su momento. Esto impide analizar la percepción y el contexto histórico de aquellos precios.

**Ejemplo:** Una acción alcanzó los 1 000 € en 1999. Tras sucesivos *splits*, el precio ajustado podría mostrar solo 100 €, ocultando la magnitud de aquel hito histórico.

**Conclusión parcial:** El ajuste **sacrifica la veracidad histórica** en favor de la consistencia económica.

#### **No resulta útil para *traders* o especuladores**

---
**Descripción:** Los operadores de corto plazo requieren precios nominales exactos para establecer niveles de soporte, resistencia y puntos de entrada/salida. El ajuste altera dichos niveles, afectando la validez de sus estrategias.

**Ejemplo:** Un *trader* planifica comprar si la acción supera los 100 €.
Con precios ajustados, ese nivel puede aparecer desplazado (por ejemplo, 50 €), invalidando su estrategia técnica.

**Conclusión parcial:** El *Adjusted Close* **no es adecuado para análisis técnico ni decisiones de *trading***.

#### **Puede ocultar señales de eventos relevantes**

---
**Descripción:** Los dividendos o *splits* pueden marcar puntos de inflexión en la evolución del precio. El ajuste elimina los saltos derivados de estos eventos, enmascarando posibles patrones de cambio.

**Ejemplo:** Una empresa cotiza a 300 €, realiza un *split 3:1* y posteriormente inicia una tendencia bajista. El precio ajustado eliminará la caída inicial, dificultando la identificación del cambio de tendencia.

**Conclusión parcial:** El uso de precios ajustados **reduce la visibilidad de ciertos comportamientos del mercado** que pueden ser clave para la interpretación técnica.

#### **Puede generar confusión si se desconoce su cálculo**

---
**Descripción:** Si el analista no comprende el significado del ajuste, podría interpretar erróneamente los precios antiguos como reales. Esto puede inducir a conclusiones falsas sobre la rentabilidad o el valor histórico de una acción.

**Ejemplo:** Una acción que hoy vale 200 € aparece con un precio ajustado de 2 € hace 20 años. Sin contexto, podría pensarse que el precio se multiplicó 100 veces, cuando en realidad ese valor fue modificado por dividendos y *splits*.

**Conclusión parcial:** El *Adjusted Close* **requiere conocimiento financiero previo** para ser interpretado correctamente.

## **Conclusión General**

El uso del precio de cierre ajustado (*Adjusted Close*) ofrece una representación más precisa del **rendimiento económico real** de una acción a lo largo del tiempo, siendo especialmente valioso para análisis de largo plazo, comparaciones entre empresas y estudios de rentabilidad acumulada.

Sin embargo, su aplicación **no es universal**: al modificar los precios nominales, pierde valor para análisis técnicos, simulaciones de mercado o estudios históricos detallados.

Por tanto:

* Para **análisis financieros, comparativos o de inversión a largo plazo**, se recomienda el uso del **precio ajustado**.
* Para **estrategias de *trading* o análisis de comportamiento de mercado**, se deben emplear **precios nominales** sin ajuste.

En el contexto del proyecto *“Predicción de las acciones de BBVA y Santander mediante RNN”*, la utilización del **precio ajustado** es la opción más adecuada, ya que permite entrenar modelos basados en series temporales **coherentes, continuas y representativas de la rentabilidad real**, eliminando distorsiones derivadas de dividendos o divisiones de acciones.
