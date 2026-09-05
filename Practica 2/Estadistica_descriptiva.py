
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

games_df = pd.read_csv("dataset_limpio.csv")

print(games_df.head())

val_frec_m = dict(games_df["meta_score"].value_counts()) # Guarda los valores unicos con su frecuencia en un diccionario
moda_m = list(val_frec_m.keys())[0]
prom_m = games_df["meta_score"].sum() / games_df["meta_score"].count() # Obtiene el promedio de la columna "meta_score"
mediana_lista_m = games_df["meta_score"].to_list()
mediana_m = mediana_lista_m[int((len(mediana_lista_m)) / 2)] # Calculamos la mediana

print("\n=== META SCORE ===")
print("Promedio:", prom_m, "\nModa:", moda_m, "\nMediana:", mediana_m) # Imprimimos los valores
print("Valor minimo:", mediana_lista_m[-1], "\nValor maximo:", mediana_lista_m[0]) # Imprimimos minimo y maximo
print("Intervalo entre minimo y maximo:", mediana_lista_m[0] - mediana_lista_m[-1])

val_frec_u = dict(games_df["user_review"].value_counts()) # Guarda los valores unicos con su frecuencia en un diccionario
moda_u = list(val_frec_u.keys())[0]
prom_u = games_df["user_review"].sum() / games_df["user_review"].count() # Obtiene el promedio de la columna "user_review"
mediana_lista_u = (games_df["user_review"].dropna()).to_list()
mediana_lista_u.sort()
n = len(mediana_lista_u) / 2
if n.is_integer(): # Calculamos la mediana
    n = int(n)
    mediana_u = (mediana_lista_u[n] + mediana_lista_u[n-1]) / 2
else:
    mediana_u = mediana_lista_u[int(n)]
mediana_lista_u.sort(reverse=True)

print("\n=== USER REVIEW ===")
print("Promedio:", prom_u, "\nModa:", moda_u, "\nMediana:", mediana_u) # Imprimimos los valores
print("Valor minimo:", mediana_lista_u[-1], "\nValor maximo:", mediana_lista_u[0])
print("Intervalo entre minimo y maximo:", mediana_lista_u[0] - mediana_lista_u[-1])

"""
Investigando hay funciones en la libreria pandas para esto mismo
Son las siguientes:
    pd.Series(games_df["user_review"]).median() MEDIANA
    pd.Series(games_df["user_review"]).mode() MODA
    pd.Series(games_df["user_review"]).mean() PROMEDIO o MEDIA
"""

"""
Otras formas de hacerlo con pandas usando dfs en lugar 
de series es de la siguiente forma:
    games_df["meta_score"].describe()
    games_df["user_review"].describe()
"""

# Obtenemos los mismos datos pero ahora agrupando por plataforma
print("\n=== PLATFORM META SCORE ===")
print(games_df.groupby("platform")["meta_score"].describe())
print("\n=== PLATFORM USER REVIEW ===")
print(games_df.groupby("platform")["user_review"].describe())
