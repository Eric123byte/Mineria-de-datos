# Eric Morales Sánchez
# 2090509

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

games_df_inicio = pd.read_csv("dataset_limpio.csv")

games_df_inicio["release_date"] = pd.to_datetime(games_df_inicio["release_date"])
games_df_inicio.info()

# Creamos una columna donde agrupemos las consolas por compañia
games_df = games_df_inicio
columnas = ["name", "platform", "summary"]
for x in columnas:
    for y in games_df[x]:
        y = str(y)
        y = y.strip()
games_df["platform"][0]

mapeo = {
    " 3DS": "Nintendo",
    " DS": "Nintendo",
    " Dreamcast": "Nintendo",
    " Game Boy Advance": "Nintendo",
    " GameCube": "Nintendo",
    " Nintendo 64": "Nintendo",
    " PC": "Otro",
    " PSP": "Sony",
    " PlayStation": "Sony",
    " PlayStation 2": "Sony",
    " PlayStation 3": "Sony",
    " PlayStation 4": "Sony",
    " PlayStation 5": "Sony",
    " PlayStation Vita": "Sony",
    " Stadia": "Google",
    " Switch": "Nintendo",
    " Wii": "Nintendo",
    " Wii U": "Nintendo",
    " Xbox": "Microsoft",
    " Xbox 360": "Microsoft",
    " Xbox One": "Microsoft",
    " Xbox Series X": "Microsoft"
}

games_df["company"] = games_df["platform"].map(mapeo)
games_df.head(10)

# ========== GRAFICA - LINEA ==========
def grafica_linea(companias, colores):
    for i in range(len(companias)):
        dates = (games_df.groupby("company")["release_date"]).get_group(companias[i])
        dates = pd.to_datetime(dates)
        dates = dates[dates.dt.year >= 2005]

        result = dates.dt.year.value_counts().sort_index().reset_index()
        result.columns = ['año', 'cantidad']

        plt.plot(result.año, result.cantidad, label=companias[i], color=colores[i])

    plt.title("Juegos publicados (2005-2021)")
    plt.xlabel("Fecha")
    plt.ylabel("Cantidad")
    plt.legend()
    plt.show

# ========== GRAFICA - BARRAS ==========
com = ["Nintendo", "Sony", "Otro", "Microsoft", "Google"]
col = ["red", "blue", "gray", "purple", "orange"]

grafica_linea(com, col)

def grafica_barras(ax, companies, colors):
    val = games_df.groupby(["company", "platform"])["user_review"].mean()
    count = 0

    for y in range(2):
        for x in range(3):
            if count == 5:
                break
            datos = val[companies[count]]
            ax[x, y].bar(datos.index, datos.values, color=colors[count])
            ax[x, y].set_title(companies[count])
            ax[x, y].set_xlabel("Plataforma")
            ax[x, y].set_ylabel("Promedio User Review")
            ax[x, y].tick_params(axis="x", rotation=45)
            count += 1

    plt.tight_layout()
    plt.show()

fig, ax = plt.subplots(3, 2, figsize=(12, 10))
grafica_barras(ax, com, col)

# ========== GRAFICA - SCATTER ==========
plt.scatter(games_df["meta_score"], games_df["user_review"], s=2)
plt.title("Meta score VS User review")
plt.xlabel("Meta score")
plt.ylabel("User review")
plt.show()

# ========== GRAFICA - HISTOGRAMA ==========
valores = list(games_df["meta_score"])

plt.hist(valores)
plt.title("Distribucion de la calificacion de MetaCritic")
plt.xlabel("Calificacion")
plt.ylabel("Cantidad de juegos")
plt.show()

# ========== GRAFICA - PIE ==========
# Buscamos el año en el cual salieron más videojuegos en todas las plataformas del csv
dates = pd.to_datetime(games_df["release_date"])
best_y = (dates.dt.year.value_counts()).idxmax()
cant_best_y = (dates.dt.year.value_counts()).max()

# Ya con el año separamos las salidas por marca, así saber cual marca tuvo mas salidas de videojuegos ese año
marcas_2018 = (
    games_df[games_df["release_date"].dt.year == best_y]
    .groupby("company")
    .size()
    .reindex(games_df["company"].unique(), fill_value=0)
)

plt.pie(marcas_2018.values, labels=marcas_2018.keys(), autopct='%1.1f%%')
plt.title("'%' de juegos salidos por plataforma (2018)")
plt.show()

games_df_inicio.to_csv("dataset_limpio.csv", index=False)
