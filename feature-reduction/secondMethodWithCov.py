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
import scipy.sparse as sp
from sklearn.cross_validation import KFold
import sklearn.svm as svm
import cPickle, gzip, numpy



def is_pos_def(x):
    return np.all( sp.linalg.eigs(x) > 0)
  
def mean_eighns(x):
    s = sp.linalg.eigsh(x)
    return np.mean(np.abs(s[s>0]))


def scatter(datax):
    n_data, n_feature = datax.shape
    #print(n_data, n_feature)
    mean_vector = datax.mean(axis=0)
    print('Mean Vector:\n', mean_vector.shape)
    
    scatter_matrix = sp.coo_matrix((n_feature,n_feature))
    for i in range(n_data):
        scatter_matrix += (datax[i,:] - mean_vector).dot(
        (datax[i,:] - mean_vector).T)
    print('Scatter Matrix:\n', scatter_matrix)



def secondMethod(datax, labels, n_class, n_feature, n_data, n_reduced_param):
    matrix = sp.coo_matrix((n_feature,0), dtype=np.float64)
    values = sp.coo_matrix((1, 0), dtype=np.float64)
    ij = np.empty(0)
    # Make a list of (class, eigenvector) tuples
    for i in xrange(n_class):
            print(i)
            # eigenvectors and eigenvalues for R_ij
            cov = scatter(datax[(np.where(labels == i)[0]),:])
            print('Covariance Matrix_B positive definite:\n', is_pos_def(cov_B))
            eig_val, eig_vec =  sp.linalg.eigh(cov)
            for ev in eig_vec:
                np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

            # Make a list of (eigenvalue, eigenvector) tuples
            eig_pairs_temp = [(eig_val[k], eig_vec[:,k])
            for k in range(len(eig_val))]

            # Sort the (eigenvalue, eigenvector) tuples from high to low
            sort_temp = sorted (eig_pairs_temp, key=lambda eig_pairs_temp: eig_pairs_temp[0])
            sort_temp.reverse()
            assert len(sort_temp) == n_feature, "The matrix has lower rank."

            for k in range(len(sort_temp)):
                values = sp.append(values, sort_temp[k][0])
                matrreduced_dataxix= sp.hstack((matrix, sort_temp[k][1].reshape(n_feature,1)))
                ij = sp.append(ij, i)

            print(len(ij))
    
   
    #Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(values[k], matrix[:,k])
    for k in range(len(values))]
     
    print('Pairs W:\n', len(eig_pairs))
    #construct the W matrix for PCA, Pick the top 100 eighgenvectors
    matrix_w = np.empty((n_feature,0), dtype=np.float64)    
    while matrix_w.shape[1] < n_reduced_param :
        for i in xrange(n_class):
            start = i*784
            for k in xrange(n_feature):
                if(eig_pairs[start + k][0] > alpha and ij[start + k]==i):
                   matrix_w_temp = np.hstack((matrix_w, eig_pairs[start + k][1].reshape(n_feature,1)))
                   if(LA.matrix_rank(matrix_w_temp) == matrix_w_temp.shape[1] ):
                       #print(eig_pairs[start + k][0])
                       matrix_w_ = np.hstack((matrix_w, eig_pairs[start + k][1].reshape(n_feature,1)))
                       break
    print('Matrix W:\n', matrix_w.shape)
    return matrix_w

n_reduced_feature = 100

F1_micro = list()
F1_macro = list()

F1_micro_total = list()
F1_macro_total = list()

n_reduced_feature_param = (2, 10, 50, 100, 200, 300, 400)


#Required datasets
csr_dataset = h5py.File('csr_ml2.h5', 'r')
td_dataset = h5py.File('2d_datax.h5', 'r')

# Labels of data
labels = np.array(csr_dataset['labels'])
n_class= len(np.unique(labels))
n_data, n_feature = csr_dataset['shape']
datax = sp.csr_matrix((csr_dataset['data'], csr_dataset['indices'], csr_dataset['indptr']),
                  csr_dataset['shape'])
print('Data Shape',datax.shape)
print('Number of Classes:', n_class)


#Second Method
test_range =xrange(1000)
train_range = xrange(1000, datax.shape[0])
print(len(test_range), len(train_range))
valid_set_x = datax[test_range, :]
valid_set_y = labels[test_range]
train_set_x = datax[train_range, :]
train_set_y = labels[train_range]



for r in n_reduced_feature_param:
    matrix_w = secondMethod(train_set_x, train_set_y, n_class, n_feature, n_data, r)
    reduced_train_set_x = matrix_w.T.dot(train_set_x.T).T
    reduced_valid_set_x = matrix_w.T.dot(valid_set_x.T).T
    clf = svm.SVC(kernel='rbf')
    clf.set_params(gamma=0.1, C=1, decision_function_shape='ovo')
    clf.fit(reduced_train_set_x, train_set_y)
    predict = clf.predict(reduced_valid_set_x)      
    F1_micro_total.append(metrics.f1_score(valid_set_y, predict, average='micro').mean())
    F1_macro_total.append(metrics.f1_score(valid_set_y, predict, average='macro').mean())   
    print("R: ", r)
                           
print(F1_micro_total)
