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

print "Set Matrices"

#N=50000
N=10000
Y = batch1_labels#+batch2_labels+batch3_labels+batch4_labels+batch5_labels
#X = np.vstack([batch1_data,batch2_data, batch3_data, batch4_data, batch5_data])
#X = X.astype(np.float32)
#X = Data.ZeroMean(X,N)

print "Load Eigen parameters"

#Get Eigenvectors and Eigenvalues
#U = np.loadtxt('EigData/Eigenvectors-right-1.txt')
#S = np.loadtxt('EigData/Eigenvalues-right-1.txt')

#print "Whitening, ZCA"

#Whiten Data and do ZCA. Then normalize the data and input it to a Gaussain
#xZCAWhite=U.dot(np.diag(1./np.sqrt(S+epsilon)))
#xZCAWhite = xZCAWhite.dot(U.T)
#xZCAWhite = xZCAWhite.dot(X.T)
#xZCAWhite = xZCAWhite.T

#print "Gaussian"

#std = np.std(xZCAWhite, axis=0, ddof=1)
#avg = np.mean(xZCAWhite,axis=0)

#xZCAWhite = 1./(std*np.sqrt(2*np.pi))*np.exp(-(xZCAWhite-avg)*(xZCAWhite-avg)/(2*std*std))

#MPGaussianTrain = Tensors.GetMaxPooledTensors(pictureSize, tensorSize, xZCAWhite,numColors)
MPGaussianTrain = np.loadtxt("Input_Train1.txt")
print "Training"

#Train the SPN
start=time.time() 
i=0
SPN1 = SPN.BuildSPN(4, [2700,90,10,1],[5,2,1,2], MPGaussianTrain[0],2)
while i < N:
    SPN.ChangeInputs(SPN1, MPGaussianTrain[i])
    SPN1 = SPN.Inference(SPN1, [2700,90,10,1],[5,2,1,2],Y[i],1)
    if i%100 == 0:
        print i
    i=i+1
end=time.time()

print "Compute Training Error"

#Calculate Training Error
train_error = [0]*100
i=0
j=0
while j < 100:
    i=0
    print j*100
    while i < 100:
        SPN.ChangeInputs(SPN1, MPGaussianTrain[i])
        SPN1[3][0].compute()
        compare = [SPN1[2][0].value, SPN1[2][1].value, SPN1[2][2].value, SPN1[2][3].value, SPN1[2][4].value,
                         SPN1[2][5].value, SPN1[2][6].value, SPN1[2][7].value, SPN1[2][8].value, SPN1[2][9].value, SPN1[2][1].value]
        train_error[j] += (compare.index(max(compare))!= Y[i])
        i=i+1
    j=j+1

print sum(train_error)/N
train_error_ = np.sum(train_error)/N

print "Load test Eig parameters"

#Get Test Data Eigvalues
#vec = np.loadtxt('EigData/Eigenvectors-test-right-1.txt')
#val = np.loadtxt('EigData/Eigenvalues-test-right-1.txt')

print "Whiten/ZCA test values"

#Whiten and ZCA Test Values
test=test_batch_data
Y=test_batch_labels
#testZCAWhite=vec.dot(np.diag(1./np.sqrt(val+epsilon)))
#testZCAWhite = testZCAWhite.dot(vec.T)
#testZCAWhite = testZCAWhite.dot(test.T)
#testZCAWhite = testZCAWhite.T

#print "gaussian test values"

#std_test = np.std(testZCAWhite, axis=0, ddof=1)
#avg_test = np.mean(testZCAWhite,axis=0)

#testZCAWhite = 1./(std_test*np.sqrt(2*np.pi))*np.exp(-(testZCAWhite-avg_test)*(testZCAWhite-avg_test)/(2*std_test*std_test))

"Compute Test Error"
MPGaussianTest = np.loadtxt("Input_test.txt")

#Get Test Error
i=0
test_error = 0
while i < N_test:
    SPN.ChangeInputs(SPN1, MPGaussianTest[i])
    SPN1[3][0].compute()
    compare = [SPN1[2][0].value, SPN1[2][1].value, SPN1[2][2].value, SPN1[2][3].value, SPN1[2][4].value,
                     SPN1[2][5].value, SPN1[2][6].value, SPN1[2][7].value, SPN1[2][8].value, SPN1[2][9].value, SPN1[2][1].value]
    test_error += (compare.index(max(compare))!= Y[i])
    if i%100 == 0:
        print i
    i=i+1
test_error = test_error/N_test
print test_error
timing=start-end

f = open('ColorSoftInf-10000.txt', 'w+')
f.write('Training Error: %s\n' %train_error_)
f.write('Test Error: %s\n' %test_error)
f.write('Training Time: %s\n:' %timing)
f.close()
