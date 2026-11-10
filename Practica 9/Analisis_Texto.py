
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re

games_df = pd.read_csv("dataset_limpio.csv")

# Verificamos la lectura del archivo
print("\nPrimeros renglones del df:\n", games_df.head())

# Son palabras "genéricas" o que no aportan nada a el análisis de la nube de palabras
stopwords = set(STOPWORDS)

# Tomamos la columan de resumenes de los juegos y la volvemos una lista en minusculas
resumen = games_df["summary"].dropna().to_list()
resumen_limpio = " ".join(i.lower() for i in resumen)
resumen_limpio = re.sub(r'[^a-zA-Z\s]', ' ', resumen_limpio)

# Creamos la nube de palabras
nube = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    colormap="viridis"
).generate(resumen_limpio)

# Imprimimos la gráfica
plt.figure(figsize=(12, 6))
plt.imshow(nube, interpolation="bilinear")
plt.axis("off")
plt.show()

# Despues de ver la primer nube de palabras se agregaron otras stopwords para mejorar el resultado
stopwords.update([
    'one', 'way', 'use', 'well', 'new',
    'get', 'first', 'make', 'back', 's',
    'also', 'even', 'much', 'many',
    'now', 'around', 'every', 'come',
    'take', 'find', 'go', 'two', 'will',
    'let', 'real', 'set', 'best', 'players will'
])

resumen = games_df["summary"].dropna().to_list()
resumen_limpio = " ".join(i.lower() for i in resumen)
resumen_limpio = re.sub(r'[^a-zA-Z\s]', ' ', resumen_limpio)

# Creamos nuevamente la nube de palabras
nube = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    stopwords=stopwords,
    colormap="viridis"
).generate(resumen_limpio)

# Imprimimos la nueva nube
plt.figure(figsize=(12, 6))
plt.imshow(nube, interpolation="bilinear")
plt.axis("off")
plt.show()

# Teniendo la nube de palabras tambien vamos a hacer otros análisis
# Obtendremos la cantidad de palabras en cada resumen y
# haremos estadística desriptiva.
df = games_df[["name", "summary"]].dropna()
df["len"] = df["summary"].str.split().str.len()
df = df.sort_values(by="len", ascending=False)
print("\nAnalisis descriptivo de la cantidad de palabras por resumen: ")
print(df["len"].describe())

# Imprimimos un histograma sobre la longitud de los resumenes
plt.hist(df["len"], bins=50)
plt.title("Distribucion de la longitud de texto")
plt.xlabel("Cantidad de palabras en resumen")
plt.ylabel("Frecuencia")
plt.show()

# Además, obtendremos la cantidad de palabras únicas que hay en total
palabras = resumen_limpio.split()
palabras_unicas = set(palabras)
d = (len(palabras_unicas) / len(palabras)) * 100
print(f"\nDiversidad del vocabulario: {d:.4f}%")
