# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:39:56 2023

@author: ALIENWARE-CERDAS
"""
def logistics(t, a, k, t0):
#""" Computes logistics function with scale and incr as free parameters
  f = a / (1.0 + np.exp(-k * (t - t0)))
  return f
def poly(t, c0, c1, c2, c3):
#""" Computes a polynominal c0 + c1*t + c2*t^2 + c3*t^3"""
  t = t - 1950
  f = c0 + c1*t + c2*t**2 + c3*t**3
  return f
import cluster_tools as ct
import numpy as np
import pandas as pd
import errors as err
import scipy.optimize as opt
import sklearn.cluster as cluster
from scipy.optimize import curve_fit
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
dataset = 'API_climatechange.csv'
data = pd.read_csv(dataset)
cereal_df = data[data['Indicator Name'].str.contains('Cereal yield')]
cereal_df = cereal_df.reset_index()
df_cereal = cereal_df[["1970", "1980", "1990","1995", "2000","2005", "2010", "2015","2017"]]
print(df_cereal.describe())
corr = df_cereal.corr()
print(corr)
ct.map_corr(df_cereal)
plt.show()
pd.plotting.scatter_matrix(df_cereal, figsize=(12, 12), s=5, alpha=0.8)
plt.show()
df_ex = df_cereal[["1995", "2000"]] # extract the two columns for clustering
df_ex = df_ex.dropna() # entries with one nan are useless
df_ex = df_ex.reset_index()
df_ex = df_ex.drop("index", axis=1)
print(df_ex.iloc[0:15])
# normalise, store minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ex)
# from sklearn import cluster

for ncluster in range(2, 10):
  # set up the clusterer with the number of expected clusters
  kmeans = cluster.KMeans(n_clusters=ncluster)
  # Fit the data, results are stored in the kmeans object
  kmeans.fit(df_norm) # fit done on x,y pairs
  labels = kmeans.labels_
  # extract the estimated cluster centres
  cen = kmeans.cluster_centers_
# calculate the silhoutte score
  print(ncluster, skmet.silhouette_score(df_ex, labels))
ncluster = 7 # best number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1995"], df_norm["2000"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("Cereal yield(1995)")
plt.ylabel("Cereal yield(2000)")
plt.show()
# The Crop yield of Sweden durinf the era from 1961 till 2021
swedendf = data[data['Country Name']== 'Sweden']
swed = swedendf[swedendf['Indicator Name']=='Cereal yield (kg per hectare)'].T
swed = swed.reset_index()
swed =swed.iloc[5:]
swed = swed.rename(columns={'index': 'date',17011: 'cereal'})
swed['date'] = swed['date'].astype(int)
swed['cereal'] = swed['cereal'].astype(int)
#print the updated dataframe
print(swed)
plt.figure()
plt.plot(swed["date"], swed["cereal"], label="data")
x_values = swed['date'].values
subset_dates = x_values[::5]  # Select every 10th date or adjust the subset as per your requirement
plt.xticks(subset_dates, rotation=45)
plt.title("Cereal yield in Sweden")
print(subset_dates)
plt.show()
