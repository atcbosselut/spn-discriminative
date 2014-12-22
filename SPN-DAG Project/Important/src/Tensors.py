from __future__ import division
import numpy as np
import random 

class Tensor:
    def __init__(self,mySize, leftCorner, pictureSize, picture, myColor):
        self.size = mySize
        self.color = myColor
        self.tensor = []
        i = 0
        while (i < self.size):
            self.tensor.append(picture[(i*pictureSize+leftCorner):(i*pictureSize+leftCorner+self.size)])
            i = i+1
        self.tensor = np.append([],self.tensor)
    
    def seeTensor(self):
        return np.array(self.tensor.reshape(self.size,self.size))

class ListOfTensors:
    def __init__(self,myPictureSize, myTensorSize, myPicture, myNumDim):
        self.numDim = myNumDim
        self.picture = myPicture
        self.numTensors = (myPictureSize-myTensorSize)*(myPictureSize*myTensorSize)
        self.pictureSize = myPictureSize
        self.tensorSize = myTensorSize
        i = 0
        self.tensorList = []
        while i < self.numDim:
            j = 0
            while (j <= self.pictureSize-self.tensorSize):
                k = 0
                while (k <= self.pictureSize-self.tensorSize):
                    self.tensorList.append(Tensor(self.tensorSize, i*self.pictureSize*self.pictureSize+j*self.pictureSize+k, 
                                                  self.pictureSize, self.picture, i))
                    k = k+1
                j = j+1
            i = i+1
            
    def seeTensorList(self):
        a = []
        for b in self.tensorList:
            a.append(b.seeTensor())
        return a

def MaxPool(tensors):
    b = tensors
    b.tensorSize = 1
    a = b.tensorList
    i=0
    while i < len(a):
        a[i] = np.max(a[i].seeTensor())
        i=i+1
    b.tensorList = a
    return b

def GetMaxPooledTensors(pictureSize, tensorSize, pictures, numColors):
    a = []
    for b in pictures:
        a.append(MaxPool(ListOfTensors(pictureSize, tensorSize, b, numColors)).tensorList)
    return a
