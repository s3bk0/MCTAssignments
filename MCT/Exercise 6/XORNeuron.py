import numpy as np
import matplotlib.pyplot as plt

def activation(x, k=1):
    return 1/(1+np.exp(-k*x))

def feedForward(input, weights):
    """propagate inputs one layer down

    Parameters
    ----------
    input : J dim vector of inputs
    weights : IxJ dim matrix of weights
    
    Returns
    -------
    I dim vector of outputs modified with activation function
    """
    return activation( np.sum(input * weights, axis=-1))

def Neuron(points, weights):
    return np.heaviside( np.sum( points * weights, axis=-1 ), 0 )

# inputs 
x0 = np.ones(4)
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

# x_i values stacked along horizontal axis -> analoguous to table
topneurons = np.stack([x0, x1, x2], axis=1)
hidneurons = np.zeros((4, 4))
outneuron = np.zeros((4, 1))

weightsl1 = np.random.rand(3, 3)
weightsl1[:-1,-1] = 0
weightsl1[[0,2], 1] = 0
weightsl2 = np.random.rand(4, 1)

# feed data forward
hidneurons[:] = feedForward(topneurons, weightsl1)
print(hidneurons)