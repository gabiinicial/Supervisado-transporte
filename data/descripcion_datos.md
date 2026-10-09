# Descripción de los datos

**Proyecto:** Predicción del nivel de ocupación de estaciones de TransMilenio (Bogotá) con árboles de decisión
**Actividad 5:** Métodos supervisados

---

## 1. Problema que se quiere resolver

En las actividades anteriores construimos un sistema basado en reglas que busca la mejor ruta entre un punto A y un punto B del sistema TransMilenio con búsqueda heurística (A\*). Ese buscador usaba costos fijos (minutos por tramo). En la realidad, un mismo tramo no "cuesta" lo mismo a las 7:00 a. m. que a las 10:00 p. m.: en hora pico las plataformas se saturan, hay que dejar pasar buses y el tiempo de viaje aumenta.

En esta actividad formulamos un **problema de aprendizaje supervisado de clasificación**:

> Dada una estación, una fecha y una hora (más el clima, los eventos y la oferta de buses), **predecir el nivel de ocupación** de la estación: `Bajo`, `Medio`, `Alto` o `Crítico`.

Formalmente, a partir de un conjunto de entrenamiento $S = \{(x_i, y_i)\}_{i=1}^{N}$, donde $x_i \in X$ es el vector de atributos y $y_i \in Y = \{\text{Bajo}, \text{Medio}, \text{Alto}, \text{Crítico}\}$ es la clase, se aprende una hipótesis $h: X \rightarrow Y$ (un árbol de decisión) que minimice el error de clasificación sobre ejemplos nuevos.

La predicción se usa como **factor de penalización** del costo de cada tramo en el buscador A\* de la Actividad 2:

$$\text{costo}(n, n') = \text{tiempo}(n, n') \times \text{factor}(\hat{y})$$

| Nivel predicho $\hat{y}$ | Bajo | Medio | Alto | Crítico |
|---|---|---|---|---|
| Factor | 1.00 | 1.10 | 1.30 | 1.60 |

---

## 2. Fuentes de datos identificadas

| # | Fuente | Entidad | Contenido útil | Enlace |
|---|---|---|---|---|
| 1 | Portal de Datos Abiertos de TransMilenio | TransMilenio S.A. | Validaciones (entradas) por estación troncal en intervalos de 15 min; salidas; ubicación de estaciones y troncales; rutas troncales y zonales | https://datosabiertos-transmilenio.hub.arcgis.com/ |
| 2 | Datos Abiertos Bogotá | Alcaldía Mayor de Bogotá | Capas geográficas del sistema, estaciones, demanda agregada | https://datosabiertos.bogota.gov.co/ |
| 3 | Portal de Datos Abiertos de Colombia | MinTIC | Conjuntos de datos de transporte público y movilidad | https://www.datos.gov.co/ |
| 4 | SIMUR - Sistema Integrado de Información sobre Movilidad Urbana | Secretaría Distrital de Movilidad | Encuesta de Movilidad 2023, aforos, velocidades | https://www.simur.gov.co/ |
| 5 | DHIME - Consulta de datos hidrometeorológicos | IDEAM | Precipitación horaria de estaciones meteorológicas de Bogotá | http://dhime.ideam.gov.co/ |
| 6 | Calendario de festivos de Colombia | Ley 51 de 1983 ("Ley Emiliani") | Días festivos que cambian la demanda | — |

### ¿Por qué se generó una muestra de datos?

Las fuentes públicas tienen la **demanda** (validaciones de entrada) pero no publican, en un mismo archivo y con la misma granularidad, la **oferta** de buses por estación y hora, la ocupación de plataformas, la lluvia y los eventos masivos. Además, los archivos de validaciones tienen millones de registros por mes, lo que no es práctico para un ejercicio académico.

Por eso, siguiendo la indicación de la actividad (*"En caso de no existir dichas fuentes de datos, desarrolle un dataset con una muestra de dichos datos"*), construimos una **muestra sintética reproducible** con el script [`src/generar_dataset.py`](../src/generar_dataset.py). La muestra usa **nombres, troncales y tipos de estación reales** y reproduce los patrones que muestran las fuentes 1, 4 y 5:

- demanda **bimodal** con picos a las 6:30 a. m. y a las 5:45 p. m.;
- los **portales** (periferia) tienen más entradas en la mañana y las estaciones del **centro** en la tarde;
- los sábados la demanda baja aprox. al 55 % y los domingos/festivos al 38 %;
- en Bogotá la lluvia suele caer en la tarde y aumenta el uso del sistema;
- los eventos en El Campín / Movistar Arena saturan las estaciones cercanas en la noche.

---

## 3. Archivos

| Archivo | Registros | Descripción |
|---|---|---|
| [`dataset_transporte.csv`](dataset_transporte.csv) | 33 600 | Una fila por **estación × día × hora**. 30 estaciones × 56 días (3 feb – 30 mar 2025) × 20 horas de operación (4:00 a 23:00). |
| [`estaciones_transmilenio.csv`](estaciones_transmilenio.csv) | 30 | Catálogo de estaciones: troncal, tipo, zona y número de conexiones. |

Para regenerarlos: `python src/generar_dataset.py` (semilla 42 → siempre los mismos datos).

---

## 4. Diccionario de datos — `dataset_transporte.csv`

| Columna | Tipo | Valores / rango | Descripción | ¿Entra al modelo? |
|---|---|---|---|---|
| `fecha` | fecha | 2025-02-03 … 2025-03-30 | Día de operación | No (identificador) |
| `dia_semana` | entero | 0 = lunes … 6 = domingo | Día de la semana | **Sí** |
| `nombre_dia` | texto | Lunes … Domingo | Nombre del día (legibilidad) | No (redundante) |
| `tipo_dia` | categórica | Laboral, Sábado, Domingo/Festivo | Tipo de día operativo | **Sí** |
| `es_festivo` | binaria | 0 / 1 | 1 si es festivo (24-mar-2025, San José) | **Sí** |
| `hora` | entero | 4 … 23 | Hora de inicio del intervalo de una hora | **Sí** |
| `franja` | categórica | Madrugada (4-5), Pico AM (6-8), Valle (9-15), Pico PM (16-19), Nocturna (20-23) | Franja operativa | **Sí** |
| `estacion` | categórica | 30 estaciones | Nombre de la estación | **Sí** |
| `troncal` | categórica | A - Caracas, B - Autonorte, C - Suba, D - Calle 80, E - NQS Central, F - Américas, G - NQS Sur, H - Caracas Sur, J - Eje Ambiental, K - Calle 26, L - Carrera 10 | Troncal a la que pertenece | **Sí** |
| `tipo_estacion` | categórica | Portal, Intermedia, Sencilla | Tipo de estación | **Sí** |
| `zona` | categórica | Periferia, Corredor, Centro | Ubicación funcional en la ciudad | **Sí** |
| `conexiones` | entero | 1 … 3 | Troncales con las que conecta la estación | **Sí** |
| `lluvia_mm` | real | 0 … ≈ 25 mm/h (≈ 0,5 % faltantes) | Precipitación en la hora | **Sí** (imputada con la mediana) |
| `evento_especial` | binaria | 0 / 1 | 1 si hay evento masivo cercano (16:00-22:00) | **Sí** |
| `frecuencia_buses` | entero | ≥ 1 buses/hora | Buses programados que paran en la estación | **Sí** |
| `validaciones_entrada` | entero | pasajeros/hora | Entradas registradas (demanda) | **No — fuga** |
| `capacidad_hora` | entero | pasajeros/hora | `frecuencia_buses × 150` | **No — fuga** |
| `ocupacion_pct` | real | % | `100 × validaciones / capacidad` | **No — fuga** |
| `nivel_ocupacion` | categórica | Bajo, Medio, Alto, Crítico | **Variable objetivo** $y$ | Objetivo |

### Definición de la variable objetivo

$$\text{ocupación} = \frac{\text{validaciones\_entrada}}{\text{frecuencia\_buses} \times 150}$$

$$
y = \begin{cases}
\text{Bajo} & \text{si ocupación} < 0.45 \\
\text{Medio} & \text{si } 0.45 \le \text{ocupación} < 0.70 \\
\text{Alto} & \text{si } 0.70 \le \text{ocupación} < 0.90 \\
\text{Crítico} & \text{si ocupación} \ge 0.90
\end{cases}
$$

### Fuga de información (*data leakage*)

`validaciones_entrada`, `capacidad_hora` y `ocupacion_pct` **no se usan como entrada**, porque la clase se calcula directamente con ellas: un árbol con esas variables tendría ~100 % de exactitud sin aprender nada útil y, además, en el momento de planear un viaje esos valores todavía no se conocen. Sí se usa `frecuencia_buses`, porque es la **oferta programada** que el operador publica con anticipación.

---

## 5. Diccionario de datos — `estaciones_transmilenio.csv`

| Columna | Descripción |
|---|---|
| `id_estacion` | Código E01 … E30 |
| `estacion` | Nombre real de la estación |
| `troncal` | Troncal (letra y nombre) |
| `tipo_estacion` | Portal / Intermedia / Sencilla |
| `zona` | Periferia / Corredor / Centro |
| `conexiones` | Número de troncales con las que conecta |

---

## 6. Proceso de generación (modelo generador)

Para cada estación $e$, día $d$ y hora $h$:

$$\text{demanda}(e,d,h) = P_e \cdot \text{perfil}_{z}(h) \cdot f_{\text{día}}(d) \cdot f_{\text{lluvia}} \cdot f_{\text{evento}} \cdot \varepsilon, \qquad \varepsilon \sim \text{LogNormal}(0,\ 0.10)$$

- $P_e = 10\,000 \times \text{demanda\_base}_e$ pasajeros/hora (presión relativa de cada estación, entre 0.35 y 0.85).
- Perfil horario bimodal (suma de gaussianas):
  $\text{perfil}_z(h) = \lambda(h) + a_z\,G(h; 6.5, 1.0) + b_z\,G(h; 17.8, 1.3) + c_z\,G(h; 12.5, 2.5)$, con $G(h;\mu,\sigma) = e^{-(h-\mu)^2 / 2\sigma^2}$.
  - Periferia: $a=1.15,\ b=0.60$ · Corredor: $a=0.90,\ b=0.90$ · Centro: $a=0.55,\ b=1.15$ · Fin de semana: $(0.30, 0.40, 0.70)$.
- $f_{\text{día}}$: Laboral 1.00 (viernes tarde 1.05), Sábado 0.55, Domingo/Festivo 0.38.
- $f_{\text{lluvia}} = 1 + 0.015 \cdot \min(\text{mm}, 12)$. La lluvia ocurre el 45 % de los días, en la tarde, con intensidad $\text{Gamma}(2, 2.5)$ mm/h.
- $f_{\text{evento}} = 1.45$ en 4 estaciones cercanas a escenarios, de 16:00 a 22:00, en 6 días al azar.

La oferta: $\text{frecuencia} = \text{round}(F_e \cdot \text{oferta}(\text{franja}) \cdot \text{oferta}(\text{día}) \cdot 0.9^{[\text{lluvia}>2]})$, con oferta Pico = 1.0, Valle = 0.65, Madrugada = 0.50, Nocturna = 0.45; Sábado = 0.80, Domingo = 0.60.

Se introducen intencionalmente ≈ 0,5 % de valores faltantes en `lluvia_mm` (falla de sensor) para ejercitar la etapa de limpieza e imputación.

---

## 7. Limitaciones

- Es una **muestra sintética**: los resultados demuestran el método, no describen la operación real de TransMilenio. Con los datos reales de validaciones (fuente 1) y lluvia (fuente 5) se podría reentrenar el mismo pipeline sin cambiar el código del modelo, solo el script de carga.
- La capacidad asume 150 pasajeros por bus articulado para todas las rutas.
- La lluvia se modela igual para toda la ciudad en cada hora.
