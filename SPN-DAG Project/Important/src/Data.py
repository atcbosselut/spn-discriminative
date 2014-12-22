import cPickle
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def GetLabelsAndData(data_batch):
    y = data_batch['labels']
    X = data_batch['data']
    return y, X

def ZeroMean(X, N):
    b = np.mean(X,0)
    i=0
    while i < N:
        X[i] = X[i] - b        
        i=i+1
    return X
