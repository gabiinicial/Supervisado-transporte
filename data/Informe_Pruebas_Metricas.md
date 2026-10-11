# Informe de Pruebas y Métricas — Actividad 5: Métodos Supervisados

**Proyecto:** Sistema de Predicción de Ocupación en TransMilenio  
**Algoritmo:** Árbol de Decisión (criterio Entropía / ID3)  
**Autor:** Gabriel Páez — gpaez@datainnova.co  
**Fecha:** 10 de octubre de 2026  

---

## 1. Descripción del Problema

Se aplica un método de aprendizaje supervisado para predecir el **nivel de ocupación** de las estaciones de TransMilenio en Bogotá, dado un conjunto de variables contextuales. El resultado del clasificador alimenta directamente el factor de penalización de costos del algoritmo A\* (Actividad 2).

| Clase predicha | Factor de penalización |
|---|---|
| Bajo | 1.00 |
| Medio | 1.10 |
| Alto | 1.30 |
| Crítico | 1.60 |

**Variable objetivo:** `nivel_ocupacion` ∈ {Bajo, Medio, Alto, Crítico}

**Definición de la variable objetivo:**

$$\text{ocupación} = \frac{\text{validaciones\_entrada}}{\text{frecuencia\_buses} \times 150}$$

| Nivel | Umbral |
|---|---|
| Bajo | ocupación < 0.45 |
| Medio | 0.45 ≤ ocupación < 0.70 |
| Alto | 0.70 ≤ ocupación < 0.90 |
| Crítico | ocupación ≥ 0.90 |

---

## 2. Descripción del Dataset

| Parámetro | Valor |
|---|---|
| Total de registros | 240 |
| Variables predictoras | 6 (todas categóricas) |
| Variable objetivo | nivel_ocupacion |
| Estaciones modeladas | 30 (TransMilenio Bogotá) |
| Troncales representadas | 11 |

### 2.1 Variables del modelo

| Variable | Tipo | Valores posibles |
|---|---|---|
| `estacion` | Categórica | 30 estaciones de TransMilenio |
| `tipo_estacion` | Categórica | Portal, Intermedia, Sencilla |
| `zona` | Categórica | Periferia, Corredor, Centro |
| `franja` | Categórica | Madrugada, Pico AM, Valle, Pico PM, Nocturna |
| `tipo_dia` | Categórica | Laboral, Fin de semana |
| `lluvia` | Categórica | Si, No |

> **Nota:** Las variables `validaciones_entrada`, `capacidad_hora` y `ocupacion_pct` fueron **excluidas** del modelo para prevenir fuga de información (*data leakage*), ya que la variable objetivo se deriva directamente de ellas.

### 2.2 Distribución de clases (conjunto completo)

| Clase | Registros | Porcentaje |
|---|---|---|
| Bajo | 133 | 55.4% |
| Medio | 72 | 30.0% |
| Alto | 24 | 10.0% |
| Crítico | 11 | 4.6% |

El dataset presenta **desbalance de clases**: la clase Bajo representa más de la mitad de los registros, mientras que Crítico es la clase minoritaria con solo 4.6%.

---

## 3. Configuración del Experimento

### 3.1 División del dataset

| Conjunto | Registros | Porcentaje |
|---|---|---|
| Entrenamiento | 192 | 80% |
| Prueba | 48 | 20% |

- División estratificada para preservar la distribución de clases en ambos conjuntos.
- Semilla aleatoria: `random_state=42` (reproducibilidad garantizada).

### 3.2 Hiperparámetros del árbol de decisión

| Hiperparámetro | Valor | Justificación |
|---|---|---|
| `criterion` | entropy | Implementa el principio ID3 (Ganancia de Información) |
| `max_depth` | 4 | Evita sobreajuste; permite reglas interpretables |
| `min_samples_leaf` | 3 | Requiere soporte mínimo por hoja para generalizar |
| `random_state` | 42 | Reproducibilidad de resultados |

---

## 4. Análisis ID3: Entropía y Ganancia de Información

### 4.1 Entropía del conjunto completo

La entropía total del conjunto S se calcula como:

$$H(S) = -\sum_{c} p_c \log_2(p_c)$$

Donde *p_c* es la proporción de registros de la clase *c*.

**H(S) = 1.5291 bits** (el máximo teórico con 4 clases equiprobables sería 2.0 bits)

### 4.2 Ganancia de Información por atributo

$$G(S, A) = H(S) - \sum_{v \in \text{valores}(A)} \frac{|S_v|}{|S|} \cdot H(S_v)$$

| Atributo | Ganancia de Información | Rango |
|---|---|---|
| **franja** | **0.4064** | **1.º — elegido como raíz por ID3** |
| tipo_dia | 0.1948 | 2.º |
| lluvia | 0.0294 | 3.º |
| zona | 0.0094 | 4.º |
| tipo_estacion | 0.0027 | 5.º |

**Conclusión ID3:** El atributo `franja` (franja horaria) tiene la mayor ganancia de información (0.4064 bits), por lo que el algoritmo ID3 lo selecciona como nodo raíz del árbol. Esto es coherente con el dominio: la hora del día es el factor más determinante del nivel de ocupación del transporte público.

---

## 5. Métricas de Evaluación

### 5.1 Exactitud global (Accuracy)

$$\text{Accuracy} = \frac{\text{predicciones correctas}}{\text{total de predicciones}} = \frac{39}{48} = \mathbf{0.8125 \ (81.25\%)}$$

El modelo clasifica correctamente el 81.25% de los casos en el conjunto de prueba.

### 5.2 Reporte de clasificación por clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|---|---|---|---|---|
| Alto | 1.00 | 0.20 | 0.33 | 5 |
| Bajo | 0.87 | 0.96 | **0.91** | 27 |
| Crítico | 0.50 | 1.00 | 0.67 | 2 |
| Medio | 0.77 | 0.71 | 0.74 | 14 |
| **Macro avg** | **0.78** | **0.72** | **0.66** | **48** |
| **Weighted avg** | **0.84** | **0.81** | **0.79** | **48** |

**Definiciones:**

- **Precisión:** de todos los casos predichos como clase X, ¿cuántos realmente son X?
- **Recall:** de todos los casos reales de clase X, ¿cuántos el modelo identificó correctamente?
- **F1-Score:** media armónica entre Precisión y Recall; penaliza los casos donde alguna de las dos es baja.

### 5.3 Análisis por clase

#### Clase Bajo (mayoritaria — soporte: 27)
- **F1 = 0.91** — mejor desempeño del modelo.
- Recall de 0.96: el modelo detecta casi todos los casos de baja ocupación.
- Precisión de 0.87: la mayoría de lo que predice como Bajo efectivamente lo es.
- Solo 1 caso real Bajo fue clasificado erróneamente como Medio.

#### Clase Medio (soporte: 14)
- **F1 = 0.74** — desempeño aceptable.
- 4 de 14 casos Medio fueron clasificados como Bajo (confusión principal con la clase mayoritaria).
- Refleja la ambigüedad del umbral Bajo/Medio en horarios de transición.

#### Clase Crítico (minoritaria — soporte: 2)
- **Recall = 1.00**: el modelo detecta ambos casos Críticos del conjunto de prueba.
- **Precisión = 0.50**: también predice 2 falsos positivos Crítico (casos Alto mal clasificados).
- El F1 de 0.67 debe interpretarse con cautela dado el soporte extremadamente bajo (n=2).

#### Clase Alto (soporte: 5)
- **F1 = 0.33** — peor desempeño del modelo.
- **Recall = 0.20**: solo 1 de 5 casos Alto fue detectado correctamente.
- 2 casos Alto se clasificaron como Medio, y 2 como Crítico.
- Esta clase limita con Crítico (0.70–0.90 vs ≥ 0.90), lo que genera confusión en los bordes de decisión.

### 5.4 Matriz de confusión

|  | **Pred. Bajo** | **Pred. Medio** | **Pred. Alto** | **Pred. Crítico** |
|---|---|---|---|---|
| **Real Bajo** | **26** | 1 | 0 | 0 |
| **Real Medio** | 4 | **10** | 0 | 0 |
| **Real Alto** | 0 | 2 | **1** | 2 |
| **Real Crítico** | 0 | 0 | 0 | **2** |

**Lectura de la matriz:**
- La diagonal principal (negrita) representa predicciones correctas.
- Las celdas fuera de la diagonal son errores de clasificación.
- No hay errores entre clases no adyacentes (Bajo ↔ Crítico = 0 errores), lo que indica que el modelo respeta el orden natural de la escala de ocupación.

---

## 6. Importancia de Variables

Las importancias son calculadas por scikit-learn como la reducción total de impureza (entropía) que cada variable produce en el árbol entrenado.

| Variable | Importancia | Porcentaje |
|---|---|---|
| `franja` | 0.4844 | **48.4%** |
| `tipo_dia` | 0.3785 | **37.9%** |
| `lluvia` | 0.1048 | **10.5%** |
| `estacion` | 0.0323 | **3.2%** |
| `tipo_estacion` | 0.0000 | 0.0% |
| `zona` | 0.0000 | 0.0% |

**Observaciones:**

1. `franja` y `tipo_dia` explican conjuntamente el **86.3%** de la capacidad predictiva del modelo. El cuándo (hora + tipo de día) es mucho más determinante que el dónde (estación, zona) para predecir ocupación.
2. `lluvia` aporta un 10.5% adicional, confirmando que las condiciones climáticas afectan la demanda de transporte.
3. `tipo_estacion` y `zona` tienen importancia cero — el modelo no las utiliza. Esto sugiere que, dado que ya se conoce la franja horaria y el tipo de día, el tipo de estación y la zona geográfica no añaden poder discriminativo adicional.

El gráfico de importancias fue exportado a `results/importancia_variables.png`.

---

## 7. Reglas de Decisión Aprendidas

El árbol (depth=4) aprendió las siguientes reglas principales (en términos interpretables):

| Franja | Tipo día | Lluvia | Predicción |
|---|---|---|---|
| Madrugada o Nocturna | Cualquiera | Cualquiera | **Bajo** |
| Pico AM o Valle | Fin de semana | Cualquiera | Bajo / Medio (según estación) |
| Pico PM, Valle | Laboral | No | **Bajo** |
| Pico PM, Valle | Laboral | Sí | **Bajo** |
| Pico AM | Laboral | No | **Alto** |
| Pico AM | Laboral | Sí | **Crítico** |
| Pico PM | Laboral | No / Sí | **Medio** |

**Regla más importante (interpretación del nodo raíz):**
- Si `franja` ∈ {Madrugada, Nocturna} → Bajo (sin importar otras variables)
- Si `franja` ∈ {Pico AM, Pico PM, Valle} → se evalúa `tipo_dia`

---

## 8. Ejemplos de Predicción

| Estación | Franja | Tipo día | Lluvia | Predicción | Probabilidades |
|---|---|---|---|---|---|
| Portal Américas | Pico AM | Laboral | No | **Alto** | Alto=45%, Crítico=18%, Medio=36% |
| Portal Américas | Nocturna | Laboral | No | **Bajo** | Bajo=88%, Medio=12% |
| Av. Jiménez | Pico PM | Laboral | Sí | **Medio** | Alto=41%, Crítico=9%, Medio=50% |
| Calle 100 | Valle | Fin de semana | No | **Bajo** | Bajo=90%, Medio=10% |

---

## 9. Análisis de Limitaciones

| Limitación | Descripción | Impacto |
|---|---|---|
| Desbalance de clases | Bajo=55%, Crítico=5% | El modelo favorece la clase mayoritaria; el F1 macro (0.66) es inferior al weighted (0.79) |
| Clase Alto mal clasificada | Recall=0.20 para la clase Alto | En el contexto de A*, la confusión Alto↔Crítico impacta el factor de penalización (1.30 vs 1.60) |
| Dataset sintético | 240 registros generados | Los patrones aprendidos simplifican la variabilidad real de TransMilenio |
| Soporte bajo en prueba | Crítico tiene solo 2 casos de prueba | Las métricas de Crítico (Recall=1.0, Precisión=0.5) tienen alta varianza |
| Variables sin efecto | tipo_estacion y zona con importancia=0 | Pueden ser relevantes con más datos o features derivadas |

---

## 10. Conclusiones

1. **El modelo alcanza una exactitud del 81.25%** en el conjunto de prueba con 48 muestras, lo cual es satisfactorio para un árbol de decisión de profundidad limitada (max_depth=4) sobre un dataset pequeño.

2. **La franja horaria es el predictor dominante** (importancia=0.484), seguida del tipo de día (0.379). Juntos explican el 86.3% de la capacidad predictiva — consistente con la ganancia de información calculada mediante ID3.

3. **La implementación ID3 desde cero y scikit-learn convergen**: ambos enfoques seleccionan `franja` como atributo raíz, validando la coherencia entre la implementación manual y la librería.

4. **El árbol de profundidad 4 equilibra interpretabilidad y generalización.** Las reglas extraídas son legibles y coherentes con el conocimiento del dominio (picos de demanda en horas laborales, efecto de lluvia, baja ocupación en madrugada).

5. **La clase Alto presenta el mayor desafío** (F1=0.33), siendo confundida tanto con Medio como con Crítico. Se recomienda, en trabajo futuro, aplicar técnicas de re-muestreo (SMOTE) o ajuste de pesos de clase para mejorar el recall de clases minoritarias.

6. **Integración con A\*:** el clasificador es apto para producir estimaciones de penalización en tiempo real. Para escenarios críticos de seguridad (e.g., evacuaciones), se recomienda usar los vectores de probabilidad en lugar del label predicho, seleccionando el percentil conservador de penalización.

---

## 11. Especificaciones Técnicas

| Elemento | Valor |
|---|---|
| Lenguaje | Python 3.x |
| Librería principal | scikit-learn ≥ 1.3 |
| Otras librerías | pandas ≥ 2.0, matplotlib ≥ 3.7 |
| Script principal | `src/modelo_arbol_decision.py` |
| Gráfico generado | `results/importancia_variables.png` |
| Dataset | `data/dataset_transporte.csv` (240 registros) |
| Codificación de variables | LabelEncoder (sklearn) — orden alfabético |

---

*Informe generado a partir de la ejecución de `src/modelo_arbol_decision.py` sobre el dataset `data/dataset_transporte.csv`.*
