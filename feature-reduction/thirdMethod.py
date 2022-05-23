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



def is_pos_def(x):
    return np.all(sp.linalg.eigsh(x) > 0)
  
def mean_eighns(x):
    s = sp.linalg.eigsh(x)
    return np.mean(np.abs(s[s>0]))


def thirdMethod(datax, labels, n_class, n_feature, n_data, alpha, beta):
    matrix = np.empty((n_feature,0), dtype=np.float64)
    values = np.empty((1, 0), dtype=np.float64)
    
    for i in xrange(n_class):
        for j in xrange(i+1, n_class):
            print(i,j)
            # eigenvectors and eigenvalues for R_ij
            cov_A = np.cov(datax[(np.where(labels == i)[0]),:].T)
            cov_B = np.cov(datax[(np.where(labels == j)[0]),:].T)
            mean_B = mean_eighns(cov_B)
            #print('Covariance Matrix_B positive definite:\n', is_pos_def(cov_B))
            #print('Covariance Matrix_B mean:\n', mean_B)  
            eig_val, eig_vec =  scipy.linalg.eigh(cov_A, cov_B+beta*mean_B*np.identity(n_feature))
            #print('Eigenvalue {} from covariance matrix: {}'.format(eig_val.shape, eig_vec[len(eig_val)-1].shape))
            # instead of 'assert' because of rounding errors
            values = np.append(values, eig_val.ravel())
            matrix = np.hstack((matrix,eig_vec[:]))
            #print(matrix.shape, len(values))
            #print(40 * '-')

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(values[i], matrix[:,i])
    for i in range(len(values))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    sort = sorted (eig_pairs, key=lambda eig_pairs: eig_pairs[0])
    sort.reverse()
    
    #construct the W matrix for PCA, Pick the top 100 eighgenvectors
    matrix_w = np.empty((n_feature,0), dtype=np.float64)    
    for i in xrange(100):
        #print(sort[i][0])
        if(sort[i][0] > alpha):
            matrix_w = np.hstack((matrix_w, sort[i][1].reshape(n_feature,1)))
    print('Matrix W:\n', matrix_w.shape)
    return matrix_w




def thirdMethodNews(datax, labels):

  # Load the dataset
    test_range =xrange(1000)
    train_range = xrange(1000, len(datax))
    print(len(test_range), len(train_range))
    valid_set_x = datax[test_range, :]
    valid_set_y = labels[test_range]
    train_set_x = datax[train_range, :]
    train_set_y = labels[train_range]
    print(valid_set_x.shape, train_set_x.shape)
    n_class= len(np.unique(labels))
    n_data, n_feature = train_set_x.shape
    beta_param = (1, 2)
    alpha_param = (0.1, 1)
    F1_micro_total = list()
    F1_macro_total = list()
    for alpha in alpha_param:
      for beta in beta_param:
        print("Alpha Alpha:%.3f and Beta:%.3f" %(alpha, beta))
        matrix_w = thirdMethod(train_set_x, train_set_y, n_class, n_feature, n_data, alpha, beta)
        reduced_train_set_x = matrix_w.T.dot(train_set_x.T).T
        reduced_valid_set_x = matrix_w.T.dot(valid_set_x.T).T
        clf = svm.SVC(kernel='rbf')
        clf.set_params(gamma=0.1, C=1, decision_function_shape='ovo')
        clf.fit(reduced_train_set_x, train_set_y)
        predict = clf.predict(reduced_valid_set_x)      
        F1_micro_total.append(metrics.f1_score(valid_set_y, predict, average='micro').mean())
        F1_macro_total.append(metrics.f1_score(valid_set_y, predict, average='macro').mean())      
    print(F1_micro_total)
    print(F1_macro_total)



# Set the parameters by cross-validation
beta = 1
alpha = 0.1
n_reduced_feature = 100

F1_micro = list()
F1_macro = list()

F1_micro_total = list()
F1_macro_total = list()

beta_param = (1, 2)
alpha_param = (0.1, 1)
n_reduced_feature_param = (2, 10, 50, 100, 200, 300, 400)


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


#For Third Method
thirdMethodNews(datax.toarray(), labels)



