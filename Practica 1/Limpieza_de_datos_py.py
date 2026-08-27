
import pandas as pd # Importamos la libreria de pandas
import numpy as np # Importamos la lubreria de numpy

games_df = pd.read_csv("all_games.csv") # Leemos el csv donde estan los datos y lo guardamos en un dataframe

# Permite ver las primeras 5 lineas de nuestro dataframe
print("\n", games_df.head())

# Nos muestra informacion relevante de el dataframe como el tipo de dato,
# cantidad de valores nulos y cantidad de valores no nulos
print("\n", games_df.info())

# Cambiamos el tipo de dato de object a datetime, esto es para manejar mejor la informacion cuando se hagan análisis
games_df["release_date"] = pd.to_datetime(games_df["release_date"])

# La columna user_review tiene calificaciones de 0 a 10, sin embargo existe un valor que no es numerico
# el cual es "tbd" (to be determined), por lo que este será reemplazado por NaN para fines practicos
games_df["user_review"] = games_df["user_review"].replace("tbd", np.nan)

# Una vez teniendo solo NaN o numeros pasamos la columna a tipo de dato numérico
# cambiando la columna automáticamente a float
games_df["user_review"] = pd.to_numeric(games_df["user_review"])

print("\nCantidad de filas duplicadas:")
# Revisamos cuantas filas duplicadas tiene nuestro dataset
print(games_df.duplicated().sum())

print("\n", games_df.info())

# Se guarda el df limpio en un archivo csv
games_df.to_csv("dataset_limpio.csv", index=False)
