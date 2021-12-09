import numpy as np

def activation(x, k=1):
    return 1/(1+np.exp(-k*x))

def singlefeedForward(input, weights):
    """propagate inputs one layer down a general NN

    Parameters
    ----------
    input : J dim vector of inputs
    weights : IxJ dim matrix of weights
    
    Returns
    -------
    I dim vector of outputs modified with activation function
    """
    return activation( np.sum(input * weights, axis=-1))

def propagateInput(input, weightlist):
    neurons = input[..., None, :]
    for weights in weightlist:
        neurons = singlefeedForward(neurons, weights)

        # adding a column for the bias to the layer outputs
        neurons = np.concatenate([np.ones((*neurons.shape[:-1], 1)), neurons] , axis=-1)
    return neurons

A = np.repeat(np.arange(2) % 2, 4)
B = np.repeat(np.arange(4) % 2, 2)
C = np.arange(8) % 2
bias = np.ones(8)
samples = np.stack([bias, A, B, C], axis=-1)

target = np.array([0, 1, 1, 0, 1, 0, 0, 1])

# generate random weights
weightlist = [ np.random.rand(3, 4)*2-1, np.random.rand(2, 4)*2-1]

out = propagateInput(samples, weightlist)
print(out)