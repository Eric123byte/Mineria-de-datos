
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

games_df = pd.read_csv("dataset_limpio.csv")

# Se comparan nuestros datos con una distribucion normal en una grafica
# Esta nos permite comparar directamente nuestros datos contra los datos esperados si
# fuera una distribucion normal
slope, intercept, r = stats.probplot(games_df["meta_score"].dropna(), dist="norm", plot=plt)[1]
r_cuadrada = r*r
plt.show()
print(r_cuadrada)

# Se hace una prueba para verificar que los datos no tienen una distribucion normal
# Si el resultado estadistico es muy grande significa que no es una distribucion normal
# el p-value nos dice la probabilidad de que obtener nuestros valores extremos asumiendo una
# distribucion normal, entonces si sale <=0.5 se puede concluir que no tenemos distribucion normal
# de lo contrario no se puede afirmar ni rechazar nada.
resultado = stats.normaltest(games_df["meta_score"].dropna())
print("Estadistico:", resultado.statistic)
print("p-value:", resultado.pvalue)

# Vamos a crear unas boxplot por categoria para ver la distribucion de cada una y comprobar que son similares
# Para realizar la prueba de Kruskal-Wallis
fig, ax = plt.subplots(5, figsize=(6, 10))

datos = games_df.groupby("company")["meta_score"]
cuenta = 0

lista = list(games_df["company"].unique())

for x in lista:
    compania = (datos).get_group(x)
    ax[cuenta].boxplot(compania)
    ax[cuenta].set_title(x)
    cuenta += 1

plt.tight_layout()
plt.show()

# Para la prueba Kruskal-Wallis (se usa esta porque tenemos datos no paramétricos y ANOVA requiere que sean paramétricos)
# vamos a tomar 4 de las 5 categorías ya que, como vimos en las graficas pasadas, Google solo tiene 5 datos haciendo que 
# su distribucion sea distinta a las demás

nintendo = datos.get_group("Nintendo")
sony = datos.get_group("Sony")
microsoft = datos.get_group("Microsoft")
otros = datos.get_group("Otro")

h_statistic, p_value = stats.kruskal(nintendo, sony, microsoft, otros)

print(f"Estadistico: {h_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# H0 = Los grupos son iguales estadisticamente
# H1 = Al menos uno de los grupos es distinto
# El valor crítico que compararemos contra el p-value es de 5%, si p es menor a 5% significa
# que se rechaza H0, si es mayor o igual a 5% se toma H0
# Como el valor p que obtuvimos es igual a p=0.000 se rechaza H0, lo que significa que 
# al menos uno de los grupos es diferente de los demás.
