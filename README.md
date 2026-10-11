# Supervisado-transporte

**Actividad 5 — Métodos supervisados: Árboles de Decisión aplicados a TransMilenio (Bogotá)**

## Descripción

Modelo de aprendizaje supervisado que predice el **nivel de ocupación** de una estación de TransMilenio: `Bajo`, `Medio`, `Alto` o `Critico`, a partir de la estación, la hora, el tipo de día, la lluvia, los eventos y la frecuencia de buses.

La predicción se conecta con el buscador A* de la **Actividad 2**: el nivel de ocupación determina un factor de penalización que ajusta el costo de cada tramo en tiempo real.

## Estructura

```
Supervisado-transporte/
├── data/
│   ├── dataset_transporte.csv          ← 12 000 registros (estación × día × hora)
│   ├── estaciones_transmilenio.csv     ← catálogo de 30 estaciones reales
│   └── descripcion_datos.md
├── src/
│   └── modelo_arbol_decision.py        ← pipeline completo (10 pasos)
├── docs/
│   ├── informe_pruebas_metricas.md
│   ├── mapa_conceptual.md
│   └── guion_video.md
├── results/                            ← figuras, reglas y métricas (se generan al ejecutar)
├── models/                             ← modelo guardado .joblib (se genera al ejecutar)
└── requirements.txt
```

## Instalación

Requiere Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Ejecución

```bash
# Entrenar, evaluar y generar todos los resultados
python src/modelo_arbol_decision.py
```

El script ejecuta 10 pasos y genera en `results/`:
- `01_distribucion_clases.png` — distribución de la variable objetivo
- `02_ocupacion_por_hora.png` — perfil horario por tipo de día
- `03_curva_complejidad.png` — sobreajuste vs. profundidad del árbol
- `04_matriz_confusion.png` — matriz de confusión en el conjunto de prueba
- `05_arbol_decision.png` — visualización del árbol
- `06_importancia_variables.png` — importancia de cada variable
- `07_comparacion_modelos.png` — árbol vs. otros clasificadores
- `ganancia_informacion_raiz.csv` — ganancias del ID3 propio
- `reglas_id3.txt` — reglas SI-ENTONCES del ID3 propio
- `reglas_arbol.txt` — reglas del árbol scikit-learn
- `busqueda_hiperparametros.csv` — resultados del GridSearchCV
- `comparacion_modelos.csv` — tabla comparativa de modelos
- `metricas.json` — resumen de todas las métricas

## Pasos del pipeline

| # | Descripción |
|---|---|
| 1 | Carga y Análisis Exploratorio (EDA) |
| 2 | Limpieza y selección de variables (sin fuga de información) |
| 3 | Partición estratificada 80/20 |
| 4 | **ID3 propio**: entropía H(S) y ganancia de información G(S,A) |
| 5 | Árbol de Decisión scikit-learn + GridSearchCV (k=5) |
| 6 | Curva de complejidad (sobreajuste vs. profundidad) |
| 7 | Evaluación en el conjunto de prueba |
| 8 | Extracción de reglas SI-ENTONCES e importancia de variables |
| 9 | Comparación con otros modelos supervisados |
| 10 | Guardado del modelo y escenarios de predicción (enlace con A*) |

## Conexión con la Actividad 2

La predicción del árbol se usa como factor de penalización en el buscador A*:

```
costo(tramo) = tiempo_viaje × factor(nivel_predicho)
```

| Nivel | Factor |
|---|---|
| Bajo | 1.00 |
| Medio | 1.10 |
| Alto | 1.30 |
| Critico | 1.60 |

## Referencias

- Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*, cap. 17. McGraw-Hill España.
- Benítez, R. et al. (2014). *Inteligencia artificial avanzada*. Editorial UOC.
- TransMilenio S.A. Portal de Datos Abiertos. https://datosabiertos-transmilenio.hub.arcgis.com/
