import os

import pandas
import pandas as pd
from sklearn import preprocessing, cluster
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

print("---------------PART I---------------------------------------------------------------")
print("Importation du données :")
os.chdir(r"C:\Users\hp\OneDrive\Bureau\project data science")
corona = pd.read_csv("corona_morocco.csv")

print(corona)
print("-------------------------------------------------------------------")
print(corona.describe())

x = corona.iloc[:, 1:].values
print("X :")
print(x)
y = corona.iloc[:, 0]
print("Y :")
print(y)
corr = corona.corr()
print("Matrice de corrolation:")
print(corr)
# Centralisation et réduction
coronacr = preprocessing.scale(x)
print("variables centrés et réduites :")
print(coronacr)
# Vecteurs et valeurs propres
eig_vals, eig_vecs = np.linalg.eig(corr)
print("Vect  propres :")
print(eig_vecs)
print("Val  propres :")
print(eig_vals)

print("---------------PART II---------------------------------------------------------------")
z = linkage(coronacr, method="ward", metric="euclidean")
dendrogram(z, labels=corona.index, orientation="left", color_threshold=27)
plt.show()

groupes_cah = fcluster(z, t=27, criterion="distance")
print(groupes_cah)
# Observations + groupes
idg = np.argsort(groupes_cah)
print(pandas.DataFrame(corona.index[idg], groupes_cah[idg]))

# =====================KMEANS AVEC DONNEE CENTRALISE REDUITE==================#

kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(coronacr)

idk = np.argsort(kmeans.labels_)
print(pandas.DataFrame(corona.index[idg], groupes_cah[idg]))
print(kmeans.transform(coronacr))
pandas.crosstab(groupes_cah, kmeans.labels_)
