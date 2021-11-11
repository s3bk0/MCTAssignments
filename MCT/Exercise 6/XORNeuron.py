import numpy as np
import matplotlib.pyplot as plt

def activation(x, k=1):
    return 1/(1+np.exp(-k*x))

def feedForward(input, weights):
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

def H3(I1, I2, weightsl1):
    return activation(weightsl1[0] + weightsl1[1]*I1 + weightsl1[2]*I2)

def O4(I1, I2, H3, weightsl2):
    return activation(weightsl2[0] + weightsl2[1]*I1 + weightsl2[2]*I2 
            + weightsl2[3]*H3)

alpha = 0.1

# inputs 
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
target = np.array([0,1,1,0])

weights = 2*np.random.rand(7) - 1

# feed data forward
h3 = H3(x1, x2, weights[:4])
output = O4(x1, x2, h3, weights[3:])

abserror = np.sum(  0.5*(np.heaviside(output, 0) -target)**2 )
conterror = np.sum(  0.5*(output -target)**2 )

while not conterror <= 0.01:
    for i in range(4):
        # input vectors per layer
        inputl1 = np.array((1, x1[i], x2[i]))
        inputl2 = np.array((1, x1[i], x2[i], h3[i]))

        # dE/dX4
        dEdX4 = (output[i]-target[i])*(1-output[i])*output[i]
        dEdX3 = weights[-1] * (1-h3[i])*h3[i] * dEdX4
        # update layer 2 weights
        weights[3:] -=  alpha* dEdX4 * inputl2
        # update layer 1 weights
        weights[:3] -= alpha* dEdX3 * inputl1

    h3 = H3(x1, x2, weights[:4])
    output = O4(x1, x2, h3, weights[3:])

    conterror = np.sum(  0.5*(output -target)**2 )
    abserror = np.sum(  0.5*(np.heaviside(output-0.5, 0) - target)**2 )


print(weights)
print(output)