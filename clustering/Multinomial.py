# -*- coding: utf-8 -*-
from __future__ import division
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

#print(datax, labels, n_data)

random.seed()

print('% 9s' % 'init'
      '    time  Macro_F1  Micro_F1  RandIndex')

def getRandomInitial(n_features, n_cluster):
  centroids = np.zeros((n_cluster, n_features))
  random.seed()
  for k in xrange(n_cluster):
    coeff = 0
    for index in xrange(n_features):
      temp  = random.randint(25, 75)
      centroids[k, index] = temp/100
      #coeff += temp
    #centroids[k, :] /= coeff
  return centroids

def MultiEM(datax, n_cluster, n_init, max_iterations):

  t = 0
  init = 0
  n_data, n_feature = datax.shape

  #Different starting point
  predict = np.zeros(n_data)
  best_predict = np.zeros(n_data)
  
  #Required parameters for E & M Step
  old_miu = np.zeros((n_cluster, n_feature))
  z = np.zeros((n_data, n_cluster))
  old_pi = np.zeros((n_cluster))
  pi = np.zeros((n_cluster))
  
  #Keeping likelihood
  best_likelihood  = 0
  likelihood = 0
  
  while init < n_init:
     print "Init: %d" % (init)
     init += 1
     miu = getRandomInitial(n_feature, n_cluster)
     for k in xrange(n_cluster):
       pi[k] = 1/n_cluster
     #print(pi)
     if best_likelihood < likelihood:
       best_predict = predict
       best_likelihood = likelihood
       
     likelihood = -10000000.0
     old_likelihood = -100000.0
     t = 0.0
     while (t < max_iterations):
 	t +=1.0
    	old_miu = miu
    	old_pi = pi
    	old_likelihood = likelihood
    	#print(old_pi)

    	#E-Step
    	temp = np.zeros((2,1))
    	for n in xrange(n_data):
          temp[1] = 0.0
          for k in xrange(n_cluster):
            #print(79 * '_')
            temp[0] = 1.0
            for i in xrange(n_feature):
              if datax[n, i]==1:
                  tmp = old_miu[k, i]
              else:
                  tmp = 1-old_miu[k, i]
              #print("%.3f %.3f %.3f   %.3f   %.3f   %.3f"
                   # % (temp[0], tmp, old_miu[k, i], datax[n, i], pow(5-old_miu[k, i], 1-datax[n, i]), pow(old_miu[k, i], datax[n, i])))
              temp[0] *= tmp
            z[n,k] = old_pi[k] * temp[0]
            temp[1] += z[n,k]
            #print "Coeff: %f   %.3f   %.3f   %.3f " % (temp[0], old_pi[k], z[n,k], temp[1])
          
          for k in xrange(n_cluster):
            if not(temp[1]==0):
              z[n, k] /= temp[1]
            else:
              print(79 * '_')

 

	likelihood = 0.0
        for n in xrange(n_data):
          predict[n] = np.argmax(z[n,:])
          for k in xrange(n_cluster):
            temp = log(old_pi[k])
            for i in xrange(n_feature):
              if(datax[n,i]==1):
                temp += log(old_miu[k, i])
              else :
                  if (1- old_miu[k, i]) > 0:
                    temp += log(1- old_miu[k, i]) 
            temp *= z[n, k]
          likelihood  += temp

        
        print "Iteration: %d" % (t-1)
    	print "theta_A = %.5f, ll = %.2f, old_ll =%.3f" % (miu[0,0], likelihood, old_likelihood)
    	if(likelihood-old_likelihood < 0.01):
          print("Here")
          #break

        #M-Step	  
    	pi = z.sum(axis=0)
    	miu = np.zeros((n_cluster, n_feature))
        #miu = decimal.Decimal(z.transpose()) * datax
        for n in xrange(n_data):
          for k in xrange(n_cluster):
            for i in xrange(n_feature):
              miu[k,i] += z[n,k] * datax[n,i]
        for k in xrange(n_cluster):
          if pi[k]!=0:
            miu[k,:] /= pi[k]
          pi[k] /= n_data
        print(pi.shape, miu.shape)
        
  
  return best_predict




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


n_data, n_feature = b_bw_datax.shape

'''
t0 = time()
estimator = BrnouliEM(b_bw_datax[1:10,:], n_cluster, 4, 10)
bench("Bernouli EM",(time() - t0), labels[1:10], estimator, n_cluster, n_data, n_feature)
'''


#Estimate for PCA-reduced data
reduced_data = PCA(n_components=100).fit_transform(datax)
#Second way of PCA
threshold = 48
n_data, n_feature = reduced_data.shape
reduced_data_bw = np.zeros((n_data,n_feature))
for k in xrange(n_data):
    for row in xrange(n_feature):
        if reduced_data[k,row] > threshold:
            reduced_data_bw[k,row] = 1
        else:
            reduced_data_bw[k,row] = 0

t0 = time()
estimator = BrnouliEM(reduced_data_bw[1:10000], n_cluster, 4, 100)
bench("Bernouli EM",(time() - t0), labels[1:10], estimator, n_cluster, n_data, n_feature)

 

