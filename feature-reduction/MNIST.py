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
    return np.all( scipy.linalg.eigvalsh(x) > 0)
  
def mean_eighns(x):
    s =  scipy.linalg.eigvalsh(x)
    return np.mean(np.abs(s[s>0]))


def visualize(datax, labels, n_class, n_feature, n_data, alpha, beta):
    titles = np.array(["R08", "R68", "R17", "R49"])
    eig_vis = np.empty((n_feature,0), dtype=np.float64)
    i_j = np.array([(0,8), (6, 8), (1 ,7), (4, 9)])
 
    for (i,j) in i_j:
            print(i,j)
            # eigenvectors and eigenvalues for R_ij
            cov_A = np.cov(datax[(np.where(labels == i)[0]),:].T)
            cov_B = np.cov(datax[(np.where(labels == j)[0]),:].T)
            mean_B = mean_eighns(cov_B)
            eig_val, eig_vec =  scipy.linalg.eigh(cov_A, cov_B+beta*mean_B*np.identity(n_feature))

            #visulaizing
            if(i==0 and j==8):
                tmp = eig_vec[:, 0].reshape(784, 1)
                eig_vis = np.hstack((eig_vis,tmp[:]))
            if(i==6 and j==8):
               tmp = eig_vec[:, 0].reshape(784, 1)
               eig_vis = np.hstack((eig_vis,tmp[:]))
            if(i==1 and j==7):
                tmp = eig_vec[:, 0].reshape(784, 1)
                eig_vis = np.hstack((eig_vis,tmp[:]))
            if(i==4 and j==9):
                tmp = eig_vec[:, 0].reshape(784, 1)
                eig_vis = np.hstack((eig_vis,tmp[:]))

    for k in range(len(titles)):                
        plt.figure(k)
        plt.imshow(eig_vis[:,k].reshape(28, 28))
    plt.show()
 
    
def thirdMethod(datax, labels, n_class, n_feature, n_data, alpha, beta):
    n_total = int (n_class*(n_class-1)/2)
    matrix = np.empty((n_feature,0), dtype=np.float64)
    values = np.empty((1, 0), dtype=np.float64)
    index = 0
    ij = np.empty(0)
    
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

            # Make a list of (eigenvalue, eigenvector) tuples
            eig_pairs_temp = [(eig_val[k], eig_vec[:,k])
            for k in range(len(eig_val))]

            # Sort the (eigenvalue, eigenvector) tuples from high to low
            sort_temp = sorted (eig_pairs_temp, key=lambda eig_pairs_temp: eig_pairs_temp[0])
            sort_temp.reverse()
            assert len(sort_temp) == n_feature, "The matrix has lower rank."
            for k in range(len(sort_temp)):
                values = np.append(values, sort_temp[k][0])
                matrix= np.hstack((matrix, sort_temp[k][1].reshape(n_feature,1)))
                ij = np.append(ij, index)
            
            index += 1
    

    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(values[k], matrix[:,k])
    for k in range(len(values))]
     
    print('Pairs W:\n', len(eig_pairs))
    #construct the W matrix for PCA, Pick the top 100 eighgenvectors
    cont = True
    matrix_w = np.empty((n_feature,0), dtype=np.float64)    
    while matrix_w.shape[1] < 100 and cont:
        #print('Matrix W:\n', matrix_w.shape)
        cont = False
        for i in xrange(n_total):
            start = i*784
            for k in xrange(n_feature):
                #print('IJ\n', ij[start + k], eig_pairs[start + k][0])
                if(eig_pairs[start + k][0] > alpha and ij[start + k]==i):
                   matrix_w_temp = np.hstack((matrix_w, eig_pairs[start + k][1].reshape(n_feature,1)))
                   if(LA.matrix_rank(matrix_w_temp) == matrix_w_temp.shape[1] ):
                       matrix_w = np.hstack((matrix_w, eig_pairs[start + k][1].reshape(n_feature,1)))
                       ij[start + k] = -1
                       cont =  True
                       break
    print('Matrix W:\n', matrix_w.shape)
    return matrix_w

'''
Result:

'''



def thirdMethodMnist():

  # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    valid_set_xx, valid_set_yy = valid_set
    train_set_xx, train_set_yy = train_set
    train_set_x = train_set_xx[0:10000]
    train_set_y = train_set_yy[0:10000]
    valid_set_x = valid_set_xx[0:10000]
    valid_set_y = valid_set_yy[0:10000]
    print(valid_set_x.shape, train_set_x.shape)
    
    n_class= len(np.unique(train_set_y))
    n_data, n_feature = train_set_x.shape
    print (n_class)
       
 
'''
[0.10640000000000001, 0.18140000000000003, 0.72240000000000004, 0.10640000000000001, 0.19139999999999999, 0.37390000000000001, 0.10640000000000001]
[0.01923355025307303, 0.10394520844874482, 0.7454024487241051, 0.01923355025307303, 0.11635936862583432, 0.33673048934620031, 0.01923355025307303]

'''

    beta_param = (0.01, 1, 10)
    alpha_param = (1, 10, 20)
    F1_micro_total = list()
    F1_macro_total = list()
    for alpha in alpha_param:
      for beta in beta_param:
        print("Alpha:%.3f and Beta:%.3f" %(alpha, beta))
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

    print(F1_micro_total)
    print(F1_macro_total)
    #Visulazie
    visualize(train_set_x, train_set_y, n_class, n_feature, n_data, 1, 10)

 




#For Mnist dataset
print(79 * '_')
print("Third Mnist Method Result")
print(79 * '_')
thirdMethodMnist()

'''
[0.10640000000000001, 0.18140000000000003, 0.72240000000000004, 0.10640000000000001, 0.19139999999999999, 0.37390000000000001, 0.10640000000000001, 0.29459999999999997]
[0.01923355025307303, 0.10394520844874482, 0.7454024487241051, 0.01923355025307303, 0.11635936862583432, 0.33673048934620031, 0.01923355025307303, 0.26642984994387936]
'''

