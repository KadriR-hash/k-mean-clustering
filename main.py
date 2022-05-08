import os
import pandas
import pandas as pd
from sklearn import preprocessing, cluster
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

print("---------------PART I---------------------------------------------------------------")
print("Importation du donnees :")
os.chdir(r"C:\Users\hp\OneDrive\Bureau\project data science")
corona = pd.read_csv("COVID-19 Coronavirus.csv")

print(corona)
print("-------------------------------------------------------------------")
print(corona.describe())
print("-------------------------------------------------------------------")

# Selection des variables
x = corona.iloc[:, 1:].values
print("X :")
print(x)
y = corona.iloc[:, 0]
print("Y :")
print(y)
corr = corona.corr()
print("Matrices de correlation:")
print(corr)

# Centralisation et reduction
coronacr = preprocessing.scale(x)
print("variables centrés et réduites :")
print(coronacr)

# Vecteurs et valeurs propres
eig_vals, eig_vecs = np.linalg.eig(corr)
print("Vect  propre :")
print(eig_vecs)
print("Val  propre :")
print(eig_vals)

print("---------------PART II---------------------------------------------------------------")
# =====================CAH==================#
# matrice des liens + affichage
# materialisation 4 classes
# z = linkage(coronacr, method="ward", metric="euclidean")
# dendrogram(z, labels=corona.index, orientation="left", color_threshold=27)
# plt.show()
# Decoupage hauteur 27
# groupes_cah = fcluster(z, t=27, criterion="distance")
# print(groupes_cah)
# tri index + Observations + groupes
# idg = np.argsort(groupes_cah)
# print(pandas.DataFrame(corona.index[idg], groupes_cah[idg]))

# =====================KMEANS AVEC DONNEE CENTRALISE REDUITE==================#

kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(coronacr)
# Tri index des groupes
idk = np.argsort(kmeans.labels_)
# Affichage observations et groupes
print(pandas.DataFrame(corona.index[idk], kmeans.labels_[idk]))
# Distances aux centres de classes
print(kmeans.transform(coronacr))
# correspondence
# pandas.crosstab(groupes_cah, kmeans.labels_)
