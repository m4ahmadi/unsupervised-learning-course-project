# -*- coding: utf-8 -*-
from __future__ import division
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
import math
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from math import exp, log
from scipy.sparse import csr_matrix
import scipy 
from sklearn.cross_validation import KFold
import sklearn.svm as svm
import cPickle, gzip, numpy


n_reduced_feature = 100

F1_micro_total = list()
F1_macro_total = list()


n_reduced_feature_param = (50, 100, 200, 300, 400)

#Required datasets
csr_dataset = h5py.File('csr_ml2.h5', 'r')
td_dataset = h5py.File('2d_datax.h5', 'r')

# Labels of data
labels = np.array(csr_dataset['labels'])
n_class= len(np.unique(labels))
n_data, n_feature = csr_dataset['shape']
datax = csr_matrix((csr_dataset['data'], csr_dataset['indices'], csr_dataset['indptr']),
                  csr_dataset['shape'])
print('Data Shape',datax.shape)
print('Number of Classes:', n_class)

#First Method
print(79 * '_')
print("First Method Result")
print(79 * '_')
test_range =xrange(1000)
train_range = xrange(1000, datax.shape[0])
print(len(test_range), len(train_range))
valid_set_x = datax[test_range, :]
valid_set_y = labels[test_range]
train_set_x = datax[train_range, :]
train_set_y = labels[train_range]



for r in n_reduced_feature_param:
    pca = PCA(n_components=r).fit(train_set_x.toarray())
    reduced_train_set_x = pca.transform(train_set_x.toarray())
    reduced_valid_set_x = pca.transform(valid_set_x.toarray())
    clf = svm.SVC(kernel='rbf')
    clf.set_params(gamma=0.1, C=1, decision_function_shape='ovo')
    clf.fit(reduced_train_set_x, train_set_y)
    predict = clf.predict(reduced_valid_set_x)
    F1_micro_total.append(metrics.f1_score(valid_set_y, predict, average='micro').mean())
    F1_macro_total.append(metrics.f1_score(valid_set_y, predict, average='macro').mean())
    print("R: ", r)

pairs = [(n_reduced_feature_param[i], F1_macro_total[i], F1_micro_total[i])
         for i in range(len(n_reduced_feature_param))]
print('% 9s' % 'n_features  Macro_F1      Micro_F1')
for i in range(len(n_reduced_feature_param)):
    print(pairs[i])



'''
_______________________________________________________________________________
First Method Result
_______________________________________________________________________________
(1000, 1919)
('R: ', 50)
('R: ', 100)
('R: ', 200)
('R: ', 300)
('R: ', 400)
n_features  Macro_F1      Micro_F1
(50, 0.77860222814370361, 0.7579999999999999)
(100, 0.81201851028049299, 0.79500000000000004)
(200, 0.82444401644660381, 0.80800000000000005)
(300, 0.83273161281068919, 0.81699999999999995)
(400, 0.83990152871313128, 0.82499999999999996)
'''
