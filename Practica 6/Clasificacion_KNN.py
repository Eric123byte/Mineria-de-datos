
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)

games_df = pd.read_csv("dataset_limpio.csv")
games_df_1 = games_df.dropna().copy()

# Convertimos la fecha en date_time y tomamos el año
games_df_1["release_year"] = pd.to_datetime(
    games_df_1["release_date"],
    errors="coerce"
).dt.year

# Crear la categoría en la que se clasifican los juegos
# segun la calificacion de los usuarios
# Baja: menor que 5
# Media: desde 5 hasta antes de 7
# Alta: desde 7 en adelante
games_df_1["review_categoria"] = pd.cut(
    games_df_1["user_review"],
    bins=[-float("inf"), 5, 7, float("inf")],
    labels=["Baja", "Media", "Alta"],
    right=False
)

# Mostramos la distribución de las categorías
print("Cantidad de videojuegos por categoría:")
print(
    games_df_1["review_categoria"]
    .value_counts()
    .reindex(["Baja", "Media", "Alta"])
)

# Graficar los valores de las categorías
games_df_1["review_categoria"].value_counts().reindex(
    ["Baja", "Media", "Alta"]
).plot(kind="bar", rot=0)

plt.title("Calificaciones de los usuarios")
plt.xlabel("Categoria de calificacion")
plt.ylabel("Cantidad de videojuegos")
plt.tight_layout()
plt.show()

# Usaremos la compañía, la plataforma y el año de lanzamiento para
# crear el modelo de clasificacion
X = games_df_1[["company", "platform", "release_year"]]
y = games_df_1["review_categoria"]

# Guardamos el 20 % de los datos para evaluar el modelo con datos no utilizados
# durante el entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Creamos las variables categóricas y escalamos el año
preprocesador = ColumnTransformer(transformers=[("categoricas", OneHotEncoder(handle_unknown="ignore"), ["company", "platform"]),
                                                ("numerica", StandardScaler(), ["release_year"])])

# Creamos un pipeline que aplique el preprocesamiento y después KNN
pipeline_knn = Pipeline([
    ("preprocesador", preprocesador),
    ("knn", KNeighborsClassifier())
])

# Evaluamos los valores de K probados anteriormente
parametros = {
    "knn__n_neighbors": [7, 11, 15, 21, 25, 31, 41, 51]
}

# Comparamos los modelos con validación cruzada
busqueda = GridSearchCV(
    estimator=pipeline_knn,
    param_grid=parametros,
    scoring="f1_macro",
    cv=5,
    n_jobs=1,
    return_train_score=True
)

# Entrenamos el modelo y buscamos el mejor usando solo los datos de entrenamiento
busqueda.fit(X_train, y_train)

print("\nMejor valor de K:",
      busqueda.best_params_["knn__n_neighbors"])

print("Mejor F1 macro en validación cruzada:",
      round(busqueda.best_score_, 4))

# Mostramos el resultado de cada valor de K
resultados = pd.DataFrame(busqueda.cv_results_)

comparacion_k = resultados[
    [
        "param_knn__n_neighbors",
        "mean_test_score",
        "std_test_score",
        "mean_train_score"
    ]
].rename(columns={
    "param_knn__n_neighbors": "K",
    "mean_test_score": "F1_macro_validacion",
    "std_test_score": "Desviacion_F1",
    "mean_train_score": "F1_macro_entrenamiento"
})

print("\nComparación de valores de K:")
print(comparacion_k.round(4).to_string(index=False))

# Graficamos el F1 macro de entrenamiento y el de validación
plt.figure(figsize=(9, 5))

plt.plot(
    comparacion_k["K"].astype(int),
    comparacion_k["F1_macro_validacion"],
    marker="o",
    label="Validación cruzada"
)

plt.plot(
    comparacion_k["K"].astype(int),
    comparacion_k["F1_macro_entrenamiento"],
    marker="s",
    label="Entrenamiento"
)

plt.xlabel("Número de vecinos")
plt.ylabel("F1 macro")
plt.title("Valor F1 macro segun el valor de k")
plt.xticks(comparacion_k["K"].astype(int))
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Tomamos el mejor modelo encontrado durante la búsqueda
modelo_knn = busqueda.best_estimator_

# Predecimos las categorías de los videojuegos de prueba
y_pred = modelo_knn.predict(X_test)

# Calculamos valores generales para comparar modelos
accuracy = accuracy_score(y_test, y_pred)

f1_macro_prueba = f1_score(
    y_test,
    y_pred,
    labels=["Baja", "Media", "Alta"],
    average="macro",
    zero_division=0
)

print("\nResultados del modelo KNN")
print("Mejor K:", busqueda.best_params_["knn__n_neighbors"])
print("Accuracy:", round(accuracy, 4))
print("Accuracy en porcentaje:", round(accuracy * 100, 2), "%")
print("F1 macro en prueba:", round(f1_macro_prueba, 4))

# Mostrar algunos datos para cada categoría
print("\nReporte de clasificación:")
print(classification_report(
    y_test,
    y_pred,
    labels=["Baja", "Media", "Alta"],
    zero_division=0
))

# Mostramos algunos valores de predicciones
comparacion = pd.DataFrame({
    "Categoria_real": y_test,
    "Categoria_predicha": y_pred
})

print("\nValores de categorías reales y predichas:")
print(comparacion.head(15))
