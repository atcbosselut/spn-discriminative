

import numpy as np
import scipy as sp
from scipy import linalg
import cPickle
import random as rd
import SPN
import Nodes
import Tensors
import Data


#Set Hyperparameters                                                                                                                                                                                          
epsilon = .00005
eta = .1
N=50000
N_test=10000
pictureSize = 32
tensorSize = 3
numColors = 3

print "Get Data"

#Unpickle Data                                                                                                                                                                                                
test_batch = Data.unpickle("cifar-10-batches-py/test_batch")

#Get Data                                                                                                                                                                                                     
test_batch_labels, test_batch_data = Data.GetLabelsAndData(test_batch)

print "Load test Eig parameters"

#Get Test Data Eigvalues                                                                                                                                                                                      
vec = np.loadtxt('EigData/Eigenvectors-test-right-1.txt')
val = np.loadtxt('EigData/Eigenvalues-test-right-1.txt')

test=test_batch_data
Y=test_batch_labels
testZCAWhite=vec.dot(np.diag(1./np.sqrt(val+epsilon)))
testZCAWhite = testZCAWhite.dot(vec.T)
testZCAWhite = testZCAWhite.dot(test.T)
testZCAWhite = testZCAWhite.T

print "gaussian test values"

std_test = np.std(testZCAWhite, axis=0, ddof=1)
avg_test = np.mean(testZCAWhite,axis=0)

testZCAWhite = 1./(std_test*np.sqrt(2*np.pi))*np.exp(-(testZCAWhite-avg_test)*(testZCAWhite-avg_test)/(2*std_test*std_test))
MPGaussianTest = Tensors.GetMaxPooledTensors(pictureSize, tensorSize, testZCAWhite,numColors)
np.savetxt("Input_test.txt",MPGaussianTest)
