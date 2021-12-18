import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import seed

"""Notice on coding style:
I prefer to write my code as efficient and reusable as possible. This means that
I use more complicated array constructions over for loops. Also, I try to document
as precise as possible.
"""

def activation(x, k=1):
    return 1/(1+np.exp(-k*x))

def singlefeedForward(input, weights):
    """propagate N input vectors down a layer of J nodes in a general NN

    Parameters
    ----------
    input : NxJ dim matrix of inputs, 
    weights : IxJ dim matrix of weights
        first index: output node
        second index: input node
    
    Returns
    -------
    NxI dim matrix of outputs modified with activation function
    """
    return activation(weights@input.T).T #activation( np.sum(input * weights, axis=-1))

def propagateInput(input, weightlist):
    """propagate a data set of N input vectors down a general NN specified
    by the given weights in

    Parameters
    ----------
    input : NxJ array
        array of N input vectors of dimension J matching the first dimension of the first
        entry of weightlist
    weightlist : list of weight matrices specifying the NN of
        each entry of the list is a 2D array with first dimension specifying the output nodes
        and second index specifying the input nodes

    Returns
    -------
    NxK dimensonal array
        output of the NN with number of nodes in the last node K
    """
    neurons = input
    for weights in weightlist:
        # adding a column for the bias to the layer outputs
        neurons = np.concatenate([np.ones((*neurons.shape[:-1], 1)), neurons] , axis=-1)

        neurons = singlefeedForward(neurons, weights)

    return neurons

def Error(input, weightlist, target):
    return np.sum((propagateInput(input, weightlist) - target)**2)
    # return np.sum(np.abs(propagateInput(input, weightlist) - target))

A = np.repeat(np.arange(2) % 2, 4)
B = np.repeat(np.arange(4) % 2, 2)
C = np.arange(8) % 2
samples = np.stack([A, B, C], axis=-1)

target = np.array([0, 1, 1, 0, 1, 0, 0, 1])[:, None]

# generate random weights
# np.random.seed(6)
ninput = 3
nhidden = ninput
nout = 1
weightlist = [ np.random.rand(nhidden, ninput+1)*2-1, np.random.rand(nout, nhidden+1)*2-1]
# weightlist = [np.array([[-0.48368234, -0.3386054 , -0.16633564, -0.01284076],
#        [-0.6731133 ,  0.39243894,  0.22712475, -0.76480927],
#        [-0.9940008 , -0.7894447 ,  0.13210493,  0.86113702]]),
#        np.array([[ 0.67066399, -0.3129032 ,  0.43416894,  0.87543524]])]
# weightlist=[np.array([[ 0.00727413,  0.62127113,  0.75050485, -0.82273128],
#        [-0.87027182, -0.81872478, -0.99601412, -0.28724133],
#        [-0.68488487,  0.75516216, -0.20821331,  0.19463295]]), 
#        np.array([[-0.01176297, -0.33154612, -0.14833177, -0.72208643]])]

out = propagateInput(samples, weightlist)
print(weightlist)


"""
alpha = 0.2
beta = 30
maxiter = 50
betastep = 1e5 / maxiter
nweights = (ninput+1)*nhidden + (nhidden+1) * nout
eqiter = nweights*100
"""

#learning rate
alpha = 0.2
beta = 40
maxiter = 700
betastep = 5e5 / maxiter
nweights = (ninput+1)*nhidden + (nhidden+1) * nout
eqiter = nweights*100

error = Error(samples, weightlist, target)
lasterror = error
errorlist = [error]
solutionfound = False

for b in range(maxiter):
    for n in range(eqiter):
        index = np.random.randint(nweights)

        # assuming nhidden == ninput
        if index < (ninput+1)*nhidden:
            nlayer = 0
        else:
            nlayer = 1
            index -= (ninput+1)*nhidden
        
        i = index // (ninput+1)
        j = index % (ninput+1)

        dweight = alpha * (2*np.random.rand(1)-1)

        # get new configuration
        newweights = [weightlist[0].copy(), weightlist[1].copy()]
        newweights[nlayer][i, j] += dweight
        newError = Error(samples, newweights, target)

        probability = np.exp(-beta* (newError - error))
        dice = np.random.rand(1)
        if newError < error or dice < probability:
            # print(f'{n}: adjusted weight {i}{j} of layer {nlayer} by {dweight}')

            weightlist = newweights
            error = newError


        errorlist.append(error)

        if np.all(np.heaviside(propagateInput(samples, weightlist)-0.5, 0) == target ):
            print("NN output: ", propagateInput(samples, weightlist).reshape(-1))
            print('target values: ', target.reshape(-1))
            print('weights: ', weightlist)
            solutionfound = True
            break
    else:
        print(f"left inner loop with beta={beta} and error {error}")


    if solutionfound:
        break
    elif abs(lasterror - error) < 0:
        # try to avoid to fall into a local minimum
        beta =  1#abs(beta - 5*betastep)
        alpha = abs(alpha - 0.1)
        eqiter += nweights
        print(f"reseted beta to beta={beta}, alpha={alpha}")
    else:
        beta += betastep

    lasterror = error

if not solutionfound:
    print("no solution was found. Consider changing parameters")

fig, ax = plt.subplots()
errorlist = np.array(errorlist)
ax.plot(errorlist)

plt.show()

        
