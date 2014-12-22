from __future__ import division
import numpy as np
import scipy as sp
from scipy import linalg
import cPickle
import random as rd
import SPN
import Nodes
import Tensors
import Data
import time
import sklearn.neighbors as sk
#Set Hyperparameters                                                                                                                                                                                          
epsilon = .00005
eta = .1
N=10000
N_test=10000
pictureSize = 32
tensorSize = 3
numColors = 3

print "Get Data"

#Unpickle Data
data_batch1 = Data.unpickle("cifar-10-batches-py/data_batch_1")
#data_batch2 = Data.unpickle("cifar-10-batches-py/data_batch_2")
#data_batch3 = Data.unpickle("cifar-10-batches-py/data_batch_3")
#data_batch4 = Data.unpickle("cifar-10-batches-py/data_batch_4")
#data_batch5 = Data.unpickle("cifar-10-batches-py/data_batch_5")
test_batch = Data.unpickle("cifar-10-batches-py/test_batch")
#Get Data
batch1_labels, batch1_data = Data.GetLabelsAndData(data_batch1)
#batch2_labels, batch2_data = Data.GetLabelsAndData(data_batch2)
#batch3_labels, batch3_data = Data.GetLabelsAndData(data_batch3)
#batch4_labels, batch4_data = Data.GetLabelsAndData(data_batch4)
#batch5_labels, batch5_data = Data.GetLabelsAndData(data_batch5)
test_batch_labels, test_batch_data = Data.GetLabelsAndData(test_batch)

MPGaussianTrain = np.loadtxt('Input_Train1.txt')
Y=batch1_labels                                          
start = time.time()
nbrs = sk.KNeighborsClassifier(n_neighbors = 10, weights = 'distance')
nbrs.fit(MPGaussianTrain,Y)
end=time.time()

Ypredtrain = nbrs.predict(MPGaussianTrain)
end=time.time()

print "Compute Training Error"

#Calculate Training Error                                                                                                                                                                                     
train_error = np.sum(np.array(Y) != np.array(Ypredtrain))/10000
print train_error

Ytest=test_batch_labels

MPGaussianTest = np.loadtxt("Input_test.txt")

"Compute Test Error"

#Get Test Error
Ypredtest = nbrs.predict(MPGaussianTest)

test_error = np.sum(np.array(Ytest) != np.array(Ypredtest))/10000
np.savetxt('YpredtestkNN.txt', Ypredtest)
print test_error
timing = end-start
f = open('10000-10NNResults.txt', 'w+')
f.write('Training Error: %s\n' %train_error)
f.write('Test Error: %s\n' %test_error)
f.write('Timing: %s\n' %timing)
f.close()
