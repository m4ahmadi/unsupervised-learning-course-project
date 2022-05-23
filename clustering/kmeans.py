# -*- coding: utf-8 -*-
from __future__ import division
import mpmath as mp
import h5py
import numpy as np
import random
from numpy import linalg as LA
import math
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import exp, log
from decimal import *
import decimal

#Required datasets
dataset = h5py.File('../mnits0t5.h5', 'r')
bw_dataset = h5py.File('2dbw_datax.h5', 'r')
td_dataset = h5py.File('2d_datax.h5', 'r')
b_bw_dataset = h5py.File('b_bw_datax.h5', 'r')

#Extract the general x values
datax = td_dataset['x']
bw_datax = bw_dataset['x']
b_bw_datax = b_bw_dataset['x']
labels = dataset['y']
n_cluster = len(np.unique(labels))


print('% 9s' % 'init'
      '    time  Macro_F1  Micro_F1  RandIndex')


def bench(name, time, labels, predict, n_cluster, n_data, n_feature):

    print(79 * '_')
    print("n_cluster %d, \t n_data %d, \t n_feature %d"
      % (n_cluster, n_data, n_feature))

 
    print('% 9s  %.2fs  %.3f   %.3f   %.3f'
          % (name, time,
             metrics.f1_score(labels, predict, average='macro'),
             metrics.f1_score(labels, predict, average='micro'),
             metrics.adjusted_rand_score(labels, predict)))

#Estimators for real data


n_data, n_feature = bw_datax.shape

t0 = time()
estimator = KMeans(init='k-means++', n_clusters=n_cluster, n_init=100).fit(bw_datax)
bench("k-means++",(time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)

t0 = time()
estimator = KMeans(init='random', n_clusters=n_cluster, n_init=20).fit(bw_datax)
bench("random", (time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)



#Estimate for PCA-reduced data

reduced_data = PCA(n_components=100).fit_transform(datax)
bw_reduced_data = PCA(n_components=100).fit_transform(bw_datax)

n_data, n_feature = bw_reduced_data.shape
t0 = time()
estimator = KMeans(init='k-means++',
                   n_clusters=n_cluster, n_init=20).fit(bw_reduced_data)
bench("PCA-k-means++", (time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)

t0 = time()
estimator = KMeans(init='random', n_clusters=n_cluster, n_init=20).fit(bw_reduced_data)
bench("PCA-random", (time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)


#Second way of PCA
threshold = 48
n_data, n_feature = reduced_data.shape
reduced_data_bw = np.zeros((n_data,n_feature))
for k in xrange(n_data):
    for row in xrange(n_feature):
        if reduced_data[k,row] > threshold:
            reduced_data_bw[k,row] = 255
        else:
            reduced_data_bw[k,row] = 0

t0 = time()
estimator = KMeans(init='k-means++', n_clusters=n_cluster, n_init=20).fit(reduced_data_bw)
bench("PCA-k-means++", (time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)

t0 = time()
estimator = KMeans(init='random', n_clusters=n_cluster, n_init=20).fit(reduced_data_bw)
bench("PCA-random", (time() - t0), labels, estimator.labels_, n_cluster, n_data, n_feature)


