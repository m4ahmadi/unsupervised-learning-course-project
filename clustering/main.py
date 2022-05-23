# Load the dataset
import numpy as np
import urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import  label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
import matplotlib.pyplot as plt

#Loading the main datasets...
import h5py
dataset = h5py.File('../mnits0t5.h5', 'r')
print(dataset.keys())


#Extract the general x values
datax = dataset['x']
datay = dataset['y']
size = datay.size
print(datax, datay, size)

            
###############################################################################
#Making the pictures Black and White
threshold = 48

plt.figure(1)
plt.imshow(datax[2,:,:])


bw_datax = np.zeros((size,28,28))
tdbw_datax = np.zeros((size,28*28))
td_datax = np.zeros((size,28*28))

for k in xrange(36017):
    for row in xrange(28):
        for column in xrange(28):
            if datax[k,row,column] > threshold:
                bw_datax[k,row,column] = 255
            else:
                bw_datax[k,row,column] = 0
    tdbw_datax[k] = bw_datax[k].ravel()
    td_datax[k] = datax[k].ravel()

#test the BW change
plt.figure(2)
print(tdbw_datax[5,:], bw_datax[5,:,:])
plt.imshow(bw_datax[2,:,:])
plt.show()               
###############################################################################
#Store BW images
f1 = h5py.File('2dbw_datax.h5', 'w')
f2 = h5py.File('2d_datax.h5', 'w')

dset = f1.create_dataset('x',data=tdbw_datax)
dset = f2.create_dataset('x',data=td_datax)

n_data, n_feature = tdbw_datax.shape
b_bw_datax = np.zeros((n_data, n_feature))
for n in xrange(n_data):
    for k in xrange(n_feature):
    if tdbw_datax[n, k]>0:
        b_bw_datax[n, k] = 1

f3 = h5py.File('b_bw_datax.h5', 'w')
dset = f1.create_dataset('x',data=b_bw_datax)



##############################################################################

dataset = h5py.File('../mnits0t5.h5', 'r')
bw_dataset = h5py.File('2dbw_datax.h5', 'r')
td_dataset = h5py.File('2d_datax.h5', 'r')

#Extract the general x values
datax = bw_dataset['x']
datay = dataset['y']
size = datay.size
print(datax, datay, size)



