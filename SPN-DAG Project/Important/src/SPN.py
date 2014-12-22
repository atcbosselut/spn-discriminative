from __future__ import division
import numpy as np
import random
import Nodes

def BuildSPN(layers, numNodesPerLayer, nodeTypePerLayer, inputNodeValues,flag,pcn=1):
    i = 1
    layer = [[]]*layers
    layer[0] = MakeNodes(0,numNodesPerLayer[0], nodeTypePerLayer, inputNodeValues)
    while i < layers:
        if i < layers - 1:
            layer[i] = MakeNodes(i, numNodesPerLayer[i], nodeTypePerLayer)
        else:
            layer[i] = [Nodes.SumNode([],[],i)] #SigmoidNode
            layer[i][0].setGradient(1,1)
        layer[i], layer[i-1] = ConnectNodes(layer[i],layer[i-1],flag,pcn)
        if nodeTypePerLayer[i] == 2:
            count = 0
            while count < numNodesPerLayer[i]:
                layer[i][count].childWeights = Nodes.normalize(layer[i][count].childWeights)
                count = count+1
        i = i+1
    j=0
    while j < len(layer[layers-2]):
        layer[layers-2][j].addChildren([Nodes.Node(layer[layers-2][j],[],layers-2,1)])
        j = j+1
    return layer


# Depending on what type of nodes a certain layer specifies and how many nodes that layer is supposed to have, create a certain number of a certain type of node

def MakeNodes(k, numNodesPerLayer, nodeTypePerLayer, inputNodeValues=[]):
    i = 0
    nodes = []
    if nodeTypePerLayer[k] == 1:
        while (i < numNodesPerLayer):
            nodes.append(Nodes.ProdNode([],[],k))
            i = i+1
    elif nodeTypePerLayer[k] == 2: 
        while (i < numNodesPerLayer):
            nodes.append(Nodes.SumNode([],[],k,))
            i = i+1
    elif nodeTypePerLayer[k] == 3:
        while (i < numNodesPerLayer):
            nodes.append(Nodes.MaxNode([],[],k))
            i = i+1
    elif nodeTypePerLayer[k] == 4:
        while (i < numNodesPerLayer):
            nodes.append(Nodes.MaxPoolNode([],[],k, inputNodeValues[i]))
            i = i+1
    else:
        while (i < numNodesPerLayer):
            nodes.append(Nodes.Node([],[],k,inputNodeValues[i]))
            i = i+1
    return nodes


# For 2 layers, connect the nodes between them in a tree pattern. Divide the number of nodes in the lower layer by the number of nodes in the upper layer. Round this quotient up and connect that number of distinct children nodes to each parent node.
def ConnectNodes(parent,children,flag,pcn):
    if len(parent)==1 or len(parent)==10:
        parent, children = DAGConnectNodes(parent,children)
    elif flag == 0:
        parent, children = FakeConnectNodes(parent,children,pcn)
    elif flag == 2:
        parent, children = ColorConnectNodes(parent,children)
    else:
        parent, children = DAGConnectNodes(parent,children)
    return parent, children


def ColorConnectNodes(parent, children):
    cutoff = np.ceil((len(children)/3)/len(parent))
    print cutoff
    diff = np.int(np.floor(len(children)/3))
    print diff
    #children[0].addParent([parent[0]])
    #children[diff].addParent([parent[0]])
    #children[2*diff].addParent([parent[0]])
    #temp = [children[0],children[diff]]#,children[2*diff]]
    temp=[]
    j=0
    i=0
    while (j < len(children)/3):
        temp.append(children[j])
        temp.append(children[j+diff])
        #temp.append(children[j+2*diff])
        children[j].addParent([parent[i]])
        children[j+diff].addParent([parent[i]])
        #children[j+2*diff].addParent([parent[i]])
        if ((j % cutoff) == cutoff-1):
            parent[i].addChildren(temp)
            temp=[]
            i = i+1
        j=j+1
    return parent, children

def FakeConnectNodes(parent, children,pcn):
    cutoff = np.ceil(len(children)/len(parent))
    children[0].addParent([parent[0]])
    temp = [children[0]]
    j=1
    i=0
    while (j < len(children)):
        temp.append(children[j])
        children[j].addParent([parent[i]])
        if ((j % cutoff) == cutoff-1):
            parent[i].addChildren(temp)
            temp=[]
            i = i+1
        j=j+1
    return parent, children

def DAGConnectNodes(parent, children):
    cutoff = np.ceil(len(children)/len(parent))
    children[0].addParent([parent[0]])
    temp = [children[0]]
    i=0
    while (i < len(parent)):
        k=0
        parent[i].addChildren(children)
        while (k < len(children)):
            children[k].addParent([parent[i]])
            k=k+1
        i=i+1
    return parent, children

# Change the Data Point at the Leaves of the SPN

def ChangeInputs(SPN,X):
    i = 0
    while i < len(X):
        SPN[0][i].value = X[i]
        i=i+1
    return SPN

# Class Inference:
# \begin{enumerate}
#     1 - Calculate the value of the SPN when all label variables are 1
#     2 - For each layer, for each node update the gradient based on what type of node it is
#     3 - Set the label values to 0 in the SPN unless the label values is that of the correct value
#     4 - Recalculate the value of the SPN when only the true value is expressed
#     5 - Repeats step 2
#     6 - Reset all label variables to 1
# \end{enumerate}

# In[271]:

def Inference(SPN,numNodesPerLayer, nodeTypePerLayer, label,eta):
    length = len(numNodesPerLayer)
    SPN[length-1][0].compute()
    c = SPN[length-1][0].value
    #print c
    i = length -1
    flag =0
    SPN[length-1][0].updateGradient(flag,c,eta)
    k=0
    #Access Label Variable in 2nd Layer child to product node and switch the non-expressed ones to 0
    while (k < numNodesPerLayer[length-2]):
        if k != label:
            b = len(SPN[length-2][k].children)
            SPN[length-2][k].children[b-1].value = 0
        k=k+1
    SPN[length-1][0].compute()
    d = SPN[length-1][0].value
    #print d
    i = length -1
    flag =1
    SPN[length-1][0].updateGradient(flag,d,eta)
    k=0
    #Switch non-expressed label variables back to 1
    while (k < numNodesPerLayer[length-2]):
        if k != label:
            b = len(SPN[length-2][k].children)
            SPN[length-2][k].children[b-1].value = 1
        k=k+1
    if 3 in nodeTypePerLayer:
        SPN = ResetMaxNodes(SPN,nodeTypePerLayer, numNodesPerLayer,eta)
    return SPN

#Reset the max nodes after each inference cycle:

def ResetMaxNodes(SPN, nodeTypePerLayer,numNodesPerLayer,eta):
    layers = [i for i, j in enumerate(nodeTypePerLayer) if j == 3]
    for a in layers:
        length = numNodesPerLayer[a]
        i = 0
        while i < length:
            change = SPN[a][i].c
            temp1 = eta/SPN[a][i].childWeights[change[0]]
            temp2 = -eta/SPN[a][i].childWeights[change[1]]
            SPN[a][i].childWeights[change[0]] += temp1
            SPN[a][i].childWeights[change[1]] += temp2
            SPN[a][i].c = [-1,-2]
            i=i+1
    return SPN
