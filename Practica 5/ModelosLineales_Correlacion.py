import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

games_df = pd.read_csv("dataset_limpio.csv")

print(games_df.info())
games_df.head()

# Creamos un segundo dataframe sin nulos
games_df2 = games_df.dropna()
games_df2.info()

# Creamos un grafico de dispersión comparando las reseñas de los usuarios con la puntuacion de MetaCritic
# para ver si existe relacion entre ambas variables
plt.scatter(games_df2["meta_score"], games_df2["user_review"], alpha=0.2)

plt.xlabel("Meta score")
plt.ylabel("User review")
plt.title("Relacion entre Meta score y User review")

plt.show()

# Se calcula la correlacion de Pearson para conocer el tipo 
# de relacion que hay entre las variables (correlacion)
# cerca de 1 = Fuerte
# cerca de 0 = Poca o nula
# cerca de -1 = Negativa fuerte
correlacion = stats.pearsonr(
    games_df2["meta_score"],
    games_df2["user_review"]
)

print("Correlación:", correlacion.statistic)

# Se obtiene un modelo lineal del tipo y=mx+b
resultado = stats.linregress(
    games_df2["meta_score"],
    games_df2["user_review"]
)

# Con el modelo creado lo usamos para predecir valores de user review y los guardamos en y_pred
y_pred = resultado.slope * games_df2["meta_score"] + resultado.intercept

# Mostramos nuevamente el grafico de dispercion pero ahora con los valores que predijo
# nuestro modelo para comparar directamente
plt.scatter(games_df2["meta_score"], games_df2["user_review"], alpha=0.2)
plt.plot(games_df2["meta_score"], y_pred)

plt.xlabel("Meta Score")
plt.ylabel("User Review")
plt.title("Regresión lineal: Meta Score vs User Review")

plt.show()

# Cuanto aumenta el user review con respecto a meta score
print("Pendiente:", resultado.slope)
# Valor estimado de user review cuando meta score es 0
print("Intercepto:", resultado.intercept)
# Nos indica el porcentaje aproximado de la variabilidad observada
# que puede ser explicada con nuestro modelo lineal
print("R^2:", resultado.rvalue ** 2)

# Obtenemos datos para entrenar y otros para hacer pruebas
# Serán 20% pruebas y 80% para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    games_df2[["meta_score"]],
    games_df2["user_review"],
    test_size=0.2,
    random_state=42
)

# Creamos el modelo
modelo = LinearRegression()

# Entrenamos el modelo
modelo.fit(X_train, y_train)

# Mostramos la R^2 de los datos, pero esta vez es de los datos que se probaron
# no del total
r2 = modelo.score(X_test, y_test)

print("R^2 de prueba:", r2)