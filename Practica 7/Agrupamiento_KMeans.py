
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

games_df = pd.read_csv("dataset_limpio.csv")

print(games_df.head())

# Scatter plot para visualizar los datos
plt.scatter(games_df["meta_score"], games_df["user_review"], alpha=0.2)
plt.xlabel("Meta score")
plt.ylabel("User review")
plt.show()

# Como los valores elegiremos las calificaciones de usuarios y de meta critics
X = (games_df[["meta_score", "user_review"]].dropna()).values

# Escalamos los valores para evitar problemas durante el procesimiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Revisamos los posibles valores de K, del 1 al 10
suma_cuadrados = []
for k in range(1, 11):
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    kmeans_temp.fit(X_scaled)
    suma_cuadrados.append(kmeans_temp.inertia_)

# Hacemos la gráfica del codo
plt.plot(range(1, 11), suma_cuadrados, marker='o')
plt.title('Método del Codo')
plt.xlabel('Valor de k')
plt.ylabel('Suma de cuadrados')
plt.show()

# Realizamos el modelo de K-Means con el valor de 5 escogido segun la grafica anterior
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=20, random_state=42)
kmeans.fit(X_scaled)

# Etiquetas de clúster para cada punto
labels = kmeans.labels_
# Coordenadas de los centroides
centroides = kmeans.cluster_centers_ 

# Imprimimos la gráfica con sus centroides
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', alpha=0.4)
plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='X', s=100, label='Centroides')
plt.title('K-Means')
plt.xlabel('Meta score, escalado')
plt.ylabel('User score, escalado')
plt.legend()
plt.show()

# Realizamos la prueba Silhouette Score para evaluar nuestro modelo
score = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score: {score:.4f}")

# Vamos a hacer la misma prueba pero ahora para todos los valores de k,
# así los podemos comparar y verificar cual nos arroja una mejor puntuación
scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores.append(silhouette_score(X_scaled, labels))

# Imprimimos la gráfica
plt.plot(k_range, scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Mejor K segun su score')
plt.show()

# Mostramos el mejor valor de k segun su score
best_k = list(k_range)[scores.index(max(scores))]
print(f"K óptimo: {best_k} (score: {max(scores):.3f})")

"""
Como se puede observar, el mejor valor que nos arrojo el análisis fue k = 2
ya que, segun los valores que tenemos podríamos separarlo principalmente en 
juegos con buena puntuacion en ambas o con mala puntuacion en ambas, como un
crecimiento lineal.
"""