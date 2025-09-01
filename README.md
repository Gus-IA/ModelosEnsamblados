# Ensemble Learning en Python con Scikit-learn y XGBoost

Este repositorio contiene un conjunto de ejemplos prácticos de **Ensemble Learning** utilizando `scikit-learn` y `xgboost`. El objetivo es ilustrar de forma visual y práctica cómo funcionan los principales métodos de ensamblado para clasificación y regresión, tales como:

- Voting Classifier (hard y soft)
- Bagging y Pasting
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost

---

## 📚 Contenido Aprendido

A lo largo del código, se exploran varios conceptos clave de Ensemble Learning:

### 🔹 1. **Voting Classifier**
Combinación de clasificadores (Regresión Logística, Random Forest, SVC) para mejorar precisión mediante:
- `voting='hard'` (mayoría simple)
- `voting='soft'` (promedio de probabilidades)

### 🔹 2. **Bagging y Árboles de Decisión**
Uso del `BaggingClassifier` con árboles de decisión:
- Comparación entre un solo árbol y un conjunto de árboles bagged
- Visualización de las fronteras de decisión

### 🔹 3. **Random Forest**
Entrenamiento de un Random Forest:
- Uso en clasificación (Iris dataset)
- Visualización de la importancia de características
- Aplicación en el dataset MNIST

### 🔹 4. **Boosting**
#### ✅ AdaBoost
- Ensamblaje secuencial de árboles débiles (`max_depth=1`)

#### ✅ Gradient Boosting
- Creación manual de modelos secuenciales con ajuste a residuos
- Uso de `GradientBoostingRegressor` de scikit-learn
- Comparación de tasas de aprendizaje
- Implementación de **early stopping** para encontrar el número óptimo de árboles

### 🔹 5. **XGBoost**
- Entrenamiento de un modelo `XGBRegressor`
- Cálculo del error cuadrático medio (MSE)

---

## 🖼️ Visualizaciones

El notebook incluye múltiples visualizaciones:
- Fronteras de decisión de clasificadores
- Importancia de características
- Predicciones acumuladas en boosting
- Evolución del error en función del número de árboles

---

## 🔧 Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt


🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
