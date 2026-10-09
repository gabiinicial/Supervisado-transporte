"""
modelo_arbol_decision.py
========================
Actividad 5 — Métodos supervisados
Árbol de Decisión para predecir el nivel de ocupación de estaciones de TransMilenio

Variable objetivo:  nivel_ocupacion → Bajo | Medio | Alto | Critico

Uso:
    python src/modelo_arbol_decision.py
"""

import io
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Forzar UTF-8 en la terminal de Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ     = Path(__file__).resolve().parents[1]
CSV_DATA = RAIZ / "data" / "dataset_transporte.csv"

OBJETIVO = "nivel_ocupacion"
CLASES   = ["Bajo", "Medio", "Alto", "Critico"]
SEMILLA  = 42


# ===========================================================================
# SECCIÓN 1 — IMPLEMENTACIÓN PROPIA DE ID3
# (Capítulo 17, Palma Méndez, 2008)
# ===========================================================================

def entropia(etiquetas):
    """
    Entropía de Shannon del conjunto S:
        H(S) = -Σ pᵢ · log₂(pᵢ)

    Mide la impureza (incertidumbre) del conjunto.
    H = 0  → conjunto puro (todas las muestras son de la misma clase)
    H = 2  → máxima incertidumbre con 4 clases equiprobables
    """
    n = len(etiquetas)
    if n == 0:
        return 0.0
    h = 0.0
    for conteo in Counter(etiquetas).values():
        p = conteo / n
        h -= p * math.log2(p)
    return h


def ganancia_informacion(filas, etiquetas, atributo):
    """
    Ganancia de información del atributo A sobre el conjunto S:
        G(S, A) = H(S) - Σ_v  |S_v|/|S| · H(S_v)

    Mide cuánta incertidumbre reduce dividir S según los valores de A.
    ID3 elige en cada nodo el atributo con mayor G(S, A).
    """
    n = len(etiquetas)
    particiones = {}
    for fila, y in zip(filas, etiquetas):
        particiones.setdefault(fila[atributo], []).append(y)
    cond = sum(len(sub) / n * entropia(sub) for sub in particiones.values())
    return entropia(etiquetas) - cond


def mostrar_ganancias(filas, etiquetas, atributos):
    """Calcula y muestra la ganancia de información de cada atributo."""
    H = entropia(etiquetas)
    print(f"\n  H(S) = {H:.4f} bits  (máx. posible con 4 clases = 2.0 bits)")
    print(f"  G(S, A) = H(S) - Σ |Sᵥ|/|S| · H(Sᵥ)\n")
    ganancias = sorted(
        [(a, ganancia_informacion(filas, etiquetas, a)) for a in atributos],
        key=lambda t: t[1], reverse=True
    )
    for atrib, g in ganancias:
        print(f"    G(S, {atrib:<18}) = {g:.4f}")
    print(f"\n  → ID3 elegiría '{ganancias[0][0]}' como atributo raíz")
    return ganancias


# ===========================================================================
# SECCIÓN 2 — PIPELINE CON SCIKIT-LEARN
# ===========================================================================

def cargar_datos():
    df = pd.read_csv(CSV_DATA, encoding="utf-8")
    print(f"Dataset cargado: {len(df)} registros, {df.shape[1]} columnas")
    print(f"Columnas: {list(df.columns)}")
    return df


def analisis_exploratorio(df):
    print("\nDistribución de la variable objetivo:")
    dist = df[OBJETIVO].value_counts().reindex(CLASES, fill_value=0)
    for clase in CLASES:
        n   = dist[clase]
        pct = 100 * n / len(df)
        bar = "█" * int(pct / 2)
        print(f"  {clase:<8}  {n:>4}  ({pct:>5.1f}%)  {bar}")


def codificar(df):
    """
    Codifica las variables categóricas a números enteros.
    Los árboles de decisión de scikit-learn requieren entradas numéricas.
    """
    columnas_cat = [c for c in df.columns if c != OBJETIVO]
    encoders = {}
    df_enc = df.copy()
    for col in columnas_cat:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    # La clase objetivo también se codifica
    le_y = LabelEncoder()
    le_y.fit(CLASES)
    df_enc[OBJETIVO] = le_y.transform(df[OBJETIVO])
    return df_enc, columnas_cat, encoders, le_y


def entrenar_y_evaluar(df_enc, columnas_cat, le_y):
    X = df_enc[columnas_cat]
    y = df_enc[OBJETIVO]

    # Partición estratificada 80 % entrenamiento / 20 % prueba
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEMILLA
    )
    print(f"\nEntrenamiento: {len(X_tr)} registros   Prueba: {len(X_te)} registros")

    # Árbol de decisión con criterio de entropía (igual que ID3)
    arbol = DecisionTreeClassifier(
        criterion="entropy",   # usa H(S) y G(S,A) — mismo principio que ID3
        max_depth=4,           # poda por profundidad para evitar sobreajuste
        min_samples_leaf=3,    # mínimo 3 ejemplos por hoja
        random_state=SEMILLA
    )
    arbol.fit(X_tr, y_tr)

    pred = arbol.predict(X_te)

    exactitud = accuracy_score(y_te, pred)
    print(f"\nExactitud en prueba : {exactitud:.4f}  ({exactitud*100:.1f}%)")

    print("\nReporte de clasificación:")
    nombres = le_y.inverse_transform(sorted(y.unique()))
    print(classification_report(y_te, pred,
                                target_names=nombres,
                                zero_division=0))

    print("Matriz de confusión (filas = real, columnas = predicho):")
    mc = confusion_matrix(y_te, pred, labels=le_y.transform(CLASES))
    header = f"{'':>8}" + "".join(f"{c:>9}" for c in CLASES)
    print(header)
    for i, clase in enumerate(CLASES):
        fila = f"{clase:<8}" + "".join(f"{mc[i,j]:>9}" for j in range(len(CLASES)))
        print(fila)

    return arbol, columnas_cat


def mostrar_reglas(arbol, columnas_cat, le_y):
    """
    Exporta el árbol como reglas SI-ENTONCES legibles.
    Cada camino raíz → hoja es una regla que el sistema aprendió.
    """
    nombres_clases = list(le_y.inverse_transform(range(len(le_y.classes_))))
    reglas = export_text(arbol, feature_names=list(columnas_cat),
                         class_names=nombres_clases)
    print("\nReglas aprendidas por el árbol (primeros 40 renglones):")
    for linea in reglas.splitlines()[:40]:
        print(" ", linea)

    print("\nImportancia de cada variable para la predicción:")
    importancias = sorted(
        zip(columnas_cat, arbol.feature_importances_),
        key=lambda t: t[1], reverse=True
    )
    for var, imp in importancias:
        bar = "█" * int(imp * 40)
        print(f"  {var:<20} {imp:.4f}  {bar}")


# ===========================================================================
# SECCIÓN 3 — PREDICCIÓN MANUAL (demo para el video)
# ===========================================================================

def predecir_caso(arbol, columnas_cat, encoders, le_y, caso):
    """
    Predice el nivel de ocupación para un caso específico.
    Útil para demostrar el modelo durante el video.
    """
    fila = {}
    for col in columnas_cat:
        le = encoders[col]
        valor = caso[col]
        if valor in le.classes_:
            fila[col] = le.transform([valor])[0]
        else:
            fila[col] = 0  # valor no visto
    X_caso = pd.DataFrame([fila])[columnas_cat]
    pred_cod = arbol.predict(X_caso)[0]
    pred_clase = le_y.inverse_transform([pred_cod])[0]
    proba = arbol.predict_proba(X_caso)[0]
    return pred_clase, dict(zip(le_y.inverse_transform(range(len(le_y.classes_))), proba))


# ===========================================================================
# MAIN
# ===========================================================================

def separador(titulo):
    print("\n" + "─" * 60)
    print(f"  {titulo}")
    print("─" * 60)


DIR_RESULTS = RAIZ / "results"


def guardar_grafica(arbol, columnas_cat):
    """Genera una sola gráfica: importancia de variables."""
    DIR_RESULTS.mkdir(exist_ok=True)
    importancias = list(zip(columnas_cat, arbol.feature_importances_))
    importancias.sort(key=lambda t: t[1])
    vars_, vals = zip(*importancias)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(vars_, vals, color="#2a78d6", height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title("Importancia de variables — Árbol de Decisión")
    ax.set_xlabel("Importancia (reducción de entropía)")
    ax.set_xlim(0, max(vals) * 1.2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    ruta = DIR_RESULTS / "importancia_variables.png"
    fig.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"\n  Gráfica guardada en results/importancia_variables.png")


def main():
    # ── 1. Cargar datos ──────────────────────────────────────────
    separador("1. CARGA Y ANÁLISIS DE DATOS")
    df = cargar_datos()
    analisis_exploratorio(df)

    # ── 2. Entropía y ganancia (ID3 propio) ──────────────────────
    separador("2. ENTROPÍA Y GANANCIA DE INFORMACIÓN (ID3 — cap. 17)")
    atributos_disc = ["franja", "tipo_dia", "lluvia", "tipo_estacion", "zona"]
    filas_dict = df[atributos_disc].to_dict("records")
    etiquetas  = df[OBJETIVO].tolist()
    mostrar_ganancias(filas_dict, etiquetas, atributos_disc)

    # ── 3. Preparación y entrenamiento ───────────────────────────
    separador("3. ÁRBOL DE DECISIÓN CON SCIKIT-LEARN")
    df_enc, columnas_cat, encoders, le_y = codificar(df)
    arbol, columnas_cat = entrenar_y_evaluar(df_enc, columnas_cat, le_y)

    # ── 4. Reglas aprendidas ─────────────────────────────────────
    separador("4. REGLAS SI-ENTONCES APRENDIDAS")
    mostrar_reglas(arbol, columnas_cat, le_y)
    guardar_grafica(arbol, columnas_cat)

    # ── 5. Predicciones de ejemplo ───────────────────────────────
    separador("5. PREDICCIONES DE EJEMPLO (demo)")
    casos = [
        {"estacion": "Portal Americas", "tipo_estacion": "Portal",
         "zona": "Periferia", "franja": "Pico AM",
         "tipo_dia": "Laboral",       "lluvia": "No"},
        {"estacion": "Portal Americas", "tipo_estacion": "Portal",
         "zona": "Periferia", "franja": "Nocturna",
         "tipo_dia": "Laboral",       "lluvia": "No"},
        {"estacion": "Av Jimenez",      "tipo_estacion": "Intermedia",
         "zona": "Centro",    "franja": "Pico PM",
         "tipo_dia": "Laboral",       "lluvia": "Si"},
        {"estacion": "Calle 100",       "tipo_estacion": "Intermedia",
         "zona": "Centro",    "franja": "Valle",
         "tipo_dia": "Fin de semana",  "lluvia": "No"},
    ]
    for caso in casos:
        pred, proba = predecir_caso(arbol, columnas_cat, encoders, le_y, caso)
        print(f"\n  Estación : {caso['estacion']}")
        print(f"  Franja   : {caso['franja']}  |  Tipo día: {caso['tipo_dia']}  |  Lluvia: {caso['lluvia']}")
        print(f"  → Nivel predicho: {pred}")
        print(f"     Probabilidades: " +
              ", ".join(f"{k}={v:.2f}" for k, v in proba.items() if v > 0.01))


if __name__ == "__main__":
    main()
