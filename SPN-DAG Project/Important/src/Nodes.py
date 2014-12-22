from __future__ import division
import numpy as np
import random 


# Class NODE defines a simple node. Its constructor takes in the values:
# \begin{enumerate}
#     Parent - a list of parents for the node
#     Children - a list of children for the node
#     Value - the value of the node
#     layer - What layer of the SPN the node is in
#     gradient - the value of the gradient of the SPN (set to 0 by default)
# \end{enumerate}
# 
# Its methods are: 
# 
# \begin{enumerate}
#     addChildren - adds the parameter list of children given to it the list of children field
#     addParent - same as addChildren but with new parents
#     compute - return its value
# \end{enumerate}

class Node:
    def __init__(self, myParent, myChildren, myLayer, myValue):
        self.parent = myParent
        self.children = myChildren
        self.value = myValue
        self.layer = myLayer
        self.gradient = 0
        self.gradient_true = 0
        
    def addChildren(self, children):
        self.children.extend(children)
            
    def addParent(self, parent):
        self.parent.extend(parent)
                  
    def compute(self):
        return self.value
    
    def updateGradient(self, flag, root,eta):
        self.gradient


# Class SumNode defines a sum node, that calculates the weighted sum of all its children as its values. Its constructor sets the values:
# \begin{enumerate}
#     Parent - a list of parents for the node
#     Children - a list of children for the node
#     Value - the value of the node (initialized to 0)
#     layer - What layer of the SPN the node is in
# \end{enumerate}
# 
# Its methods are: 
# 
# \begin{enumerate}
#     addChildren - adds the parameter list of children given to it the list of children field. Initializes random weights for those children
#     compute - return the weighted sum of its children's values
#     updateSoftGradient - Updates its children's gradient from the formula in Gens and Domingos, 2012. If the flag is 0, we assume we are in the case where all labels are set to 1, and the gradients are calculated from this configuration. If the flag is set to 1, 
#     we are in the case where only the true label is set to 1, so we compute the gradients here and reset the weights
# \end{enumerate}

class SumNode(Node):
    MyType = 2
    
    def __init__(self, myParent, myChildren, myLayer):
        Node.__init__(self,myParent,myChildren,myLayer,0)
        self.childWeights = normalize([randomFunction(1,0) for i in range(len(myChildren))])
        self.childWeightsDelta = [0]*len(myChildren)
        
    def addChildren(self,children, myChildWeights=[]):
        if not myChildWeights:
            myChildWeights = [randomFunction(1,0) for i in range(len(children))]
        Node.addChildren(self, children)
        self.childWeights.extend(myChildWeights)
        self.childWeightsDelta.extend([0]*len(children))
        
        
    def updateGradient(self,flag,root,eta):
        i = 0
        weightGradient = []
        while i < len(self.children):
            if flag == 0:
                self.children[i].gradient = self.children[i].gradient + self.childWeights[i]*self.gradient
                self.childWeightsDelta[i]= root*self.children[i].value*self.gradient
            elif flag == 1:
                self.children[i].gradient_true = self.children[i].gradient_true + self.childWeights[i]*self.gradient_true
                self.childWeightsDelta[i] = eta*(root*self.children[i].value*self.gradient_true - self.childWeightsDelta[i])
            i = i+1
        if flag == 1:
            self.childWeights = normalize(((np.array(self.childWeights) + np.array(self.childWeightsDelta))).tolist())
        i=0
        while i < len(self.children):
            self.children[i].updateGradient(flag,root,eta)
            i=i+1
        
        
    def setGradient(self, grad, grad_true):
        self.gradient = grad
        self.gradient_true = grad_true
        
    def compute(self):
        temp = 0
        for a in self.children:
            temp = temp+a.compute()*self.childWeights[self.children.index(a)]
        self.value = temp
        return temp


# Class ProdNode defines a product node, that calculates the product of all its children as its values. Its constructor sets the values:
# \begin{enumerate}
#     Parent - a list of parents for the node
#     Children - a list of children for the node
#     Value - the value of the node (initialized to 0)
#     layer - What layer of the SPN the node is in
# \end{enumerate}
# 
# Its methods are: 
# 
# \begin{enumerate}
#     compute - return the product of its children's values
#     updateSoftGradient - Updates its children's gradient from the formula in Gens and Domingos, 2012. This formula is the same regardless of whether we are doing Hard Inference or Soft Inference
# \end{enumerate}

class ProdNode(Node):
    MyType = 1

    def __init__(self, myParent, myChildren, myLayer):
        Node.__init__(self,myParent,myChildren,myLayer,1)
    
    def updateGradient(self, flag,root,eta):
        i=0
        child_values = []
        for a in self.children:
            child_values.append(a.value)
        while i < len(self.children):
            temp = child_values[:i]+child_values[i+1:]
            if flag == 0:                
                self.children[i].gradient = self.children[i].gradient + self.gradient*np.prod(np.array(temp))
            if flag == 1:
                self.children[i].gradient_true = self.children[i].gradient_true + self.gradient_true*np.prod(np.array(temp))
            i=i+1
        i=0
        while i < len(self.children):
            self.children[i].updateGradient(flag,root,eta)
            i=i+1
        
    def compute(self):
        temp = 1
        for a in self.children:
            temp = temp*a.compute()
        self.value = temp
        return temp


# Class ProdNode defines a max node, that calculates the max of all its children as its value. Its constructor sets the values:
# \begin{enumerate}
#     Parent - a list of parents for the node
#     Children - a list of children for the node
#     Value - the value of the node (initialized to 0)
#     layer - What layer of the SPN the node is in
# \end{enumerate}
# 
# Its methods are: 
# 
# \begin{enumerate}
#     compute - return the maximum of its children's values
#     updateSoftGradient - Updates its children's gradient from the formula in Gens and Domingos, 2012. This formula is the same regardless of whether we are doing Hard Inference or Soft Inference
# \end{enumerate}

class MaxNode(Node):  
    MyType = 3
    
    def __init__(self, myParent, myChildren, myLayer):
        Node.__init__(self,myParent,myChildren, myLayer,0)
        self.childWeights = normalize([randomFunction(1,0) for i in range(len(myChildren))])
        self.c = [-1,-2]

    def addChildren(self,children, myChildWeights=[]):
        if not myChildWeights:
            myChildWeights = [randomFunction(1,0) for i in range(len(children))]
        Node.addChildren(self, children)
        self.childWeights.extend(myChildWeights)
   
    #Fix MPE Inference - Only needs a few changes (i.e. making sure the right parts are visited)     
    def updateGradient(self,flag, root, eta):
        child_val = []
        for a in self.children:
            child_val.append(a.value)
        weighted_max = (np.array(child_val)*np.array(self.childWeights)).tolist()
        #Assumes only one max (can be altered to get a list of maxes if we split the gradient)
        val_max = max(weighted_max)
        index = weighted_max.index(val_max)
        self.c[flag] = index
        if flag == 0:
            self.children[index].gradient = self.children[index].gradient + self.childWeights[index]*self.gradient
        if flag == 1:
            self.children[index].gradient_true = self.children[index].gradient_true + self.childWeights[index]*self.gradient_true
        self.children[index].updateGradient(flag,root,eta)
    
    def compute(self):
        i=0
        temp=0
        while i < len(self.children):
            b = self.children[i].compute() * self.childWeights[i]
            if b > temp:
                temp = b
                self.value = temp
            i=i+1
        return self.value

# Normalize the List

def normalize(someList):
    a = (np.array(someList)/np.sum(np.array(someList))).tolist()
    return a


# Generate a random number beween the lower and upper bound passed as a parameter

def randomFunction(upper,lower):
    return random.random()*(upper-lower) +lower
