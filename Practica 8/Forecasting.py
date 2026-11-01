import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Leemos el Dataset
games_df = pd.read_csv("dataset_limpio.csv")

games_df["platform"] = games_df["platform"].str.strip()
games_df.head()

# Trabajamos con una copia del dataset
games_forecast = games_df.copy()

games_forecast["release_date"] = pd.to_datetime(games_forecast["release_date"], errors="coerce")

# Obtenemos el año
games_forecast["year"] = games_forecast["release_date"].dt.year

# Contamos los registros por año
games_year = games_forecast.groupby("year").size()

# Imprimimos la gráfica
plt.figure(figsize=(12, 5))

plt.plot(
    games_year.index,
    games_year.values,
    marker="o"
)

plt.title("Cantidad de lanzamientos de videojuegos por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de juegos")
plt.grid()

plt.show()

# Utilizamos solo los años completos
games_year_completo = games_year.loc[1995:2020]

# Separamos los datos para entrenamiento y prueba
train = games_year_completo.loc[1995:2015]
test = games_year_completo.loc[2016:2020]

# Imprimimos los resultados}
print("\nEntrenamiento:", train.index.min(), "-", train.index.max())
print("Registros de entrenamiento:", len(train))

print("\nPrueba:", test.index.min(), "-", test.index.max())
print("Registros de prueba:", len(test))

# Convertimos los índices de año a fechas completas
train_arima = train.copy()
test_arima = test.copy()

train_arima.index = pd.to_datetime(train_arima.index.astype(str), format="%Y")
test_arima.index = pd.to_datetime(test_arima.index.astype(str), format="%Y")

# Establecemos la frecuencia (Nos permite seleccionar cuanto queremos que prediga el modelo, en nuestro caso va a predecir por año)
train_arima = train_arima.asfreq("YS")
test_arima = test_arima.asfreq("YS")

# Creamos el modelo
modelo_arima = ARIMA(train_arima, order=(1, 1, 1))

# Entrenamos el modelo
resultado_arima = modelo_arima.fit()

# Hacemos las predicciones
pred_arima = resultado_arima.forecast(steps=len(test_arima))

# Evaluamos el modelo
# El MAE indica el error promedio aproximado entre la prediccion y el valor real
# Al igual que el MAE calcula el error entre el dato predicho y el real, sin embargo, este 
# aplica una penalizacion a los valores que se hayan equivocado por un mayor rango
mae_arima = mean_absolute_error(test_arima, pred_arima)
rmse_arima = np.sqrt(mean_squared_error(test_arima, pred_arima))

# Imprimimos la evaluación
print("\n========== ARIMA ==========")
print("Predicciones:", pred_arima.round(2).values)
print("MAE:", mae_arima)
print("RMSE:", rmse_arima)

# Mostramos la gráfica
plt.figure(figsize=(12, 5))

plt.plot(train_arima.index, train_arima.values, label="Entrenamiento")
plt.plot(test_arima.index, test_arima.values, marker="o", label="Datos reales")
plt.plot(pred_arima.index, pred_arima.values, marker="o",
         linestyle="--", label="ARIMA")

plt.title("ARIMA: cantidad de lanzamientos por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de videojuegos")
plt.legend()
plt.grid(True)
plt.show()

# Creamos el modelo
modelo_holt = ExponentialSmoothing(
    train_arima,
    trend="add",
    seasonal=None
)

# Entrenamos el modelo
resultado_holt = modelo_holt.fit()

# Hacemos las predicciones
pred_holt = resultado_holt.forecast(steps=len(test_arima))

# Evaluamos el modelo
mae_holt = mean_absolute_error(test_arima, pred_holt)
rmse_holt = np.sqrt(mean_squared_error(test_arima, pred_holt))

# Imprimimos los resultados
print("\n========== Holt ==========")
print("Predicciones:", pred_holt.round(2).values)
print("MAE:", mae_holt)
print("RMSE:", rmse_holt)

# Imprimimos la gráfica
plt.figure(figsize=(12, 5))

plt.plot(train_arima.index, train_arima.values, label="Entrenamiento")
plt.plot(test_arima.index, test_arima.values, marker="o", label="Datos reales")
plt.plot(pred_holt.index, pred_holt.values, marker="o",
         linestyle="--", label="Holt")

plt.title("Suavizamiento exponencial: cantidad de lanzamientos por año")
plt.xlabel("Año")
plt.ylabel("Cantidad de videojuegos")
plt.legend()
plt.grid(True)
plt.show()

# Como nuestro modelo de Holt tuvo mejor MAE (Mas pequeño) que el modelo ARIMA
# vamos a usarlo para predecir valores sobre salidas de videojuegos en los proximos
# años

games_year_final = games_year_completo.copy()

# Convertimos los años a fechas
games_year_final.index = pd.to_datetime(
    games_year_final.index.astype(str),
    format="%Y"
)

# Establecemos la frecuencia (Nos permite seleccionar cuanto queremos que prediga el modelo, en nuestro caso va a predecir por año)
games_year_final = games_year_final.asfreq("YS")

# Creamos el modelo
modelo_holt_final = ExponentialSmoothing(
    games_year_final,
    trend="add",
    seasonal=None
)

# Entrenamos el modelo
resultado_holt_final = modelo_holt_final.fit()

# Pronosticamos los siguientes 5 años
pred_futuro = resultado_holt_final.forecast(steps=5)

# Imprimimos las predicciones
print("\nPronóstico de lanzamientos:")
print(pred_futuro.round(2))

# IMprimimos la gráfica
plt.figure(figsize=(12, 5))

plt.plot(games_year_final.index, games_year_final.values,
         label="Datos reales")

plt.plot(pred_futuro.index, pred_futuro.values,
         marker="o", linestyle="--", label="Pronóstico Holt")

plt.title("Pronóstico de lanzamientos de videojuegos")
plt.xlabel("Año")
plt.ylabel("Cantidad de videojuegos")
plt.legend()
plt.grid(True)
plt.show()
