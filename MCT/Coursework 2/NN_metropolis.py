import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import seed


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
        # adding a column for the bias to the layer outputs (see pdf)
        neurons = np.concatenate([np.ones((*neurons.shape[:-1], 1)), neurons] , axis=-1)

        neurons = singlefeedForward(neurons, weights)

    return neurons

def Error(input, weightlist, target):
    return np.sum((propagateInput(input, weightlist) - target)**2)

# create truth table
# input rows are binary numbers up to 7
A = np.repeat(np.arange(2) % 2, 4)
B = np.repeat(np.arange(4) % 2, 2)
C = np.arange(8) % 2
samples = np.stack([A, B, C], axis=-1)

target = np.array([0, 1, 1, 0, 1, 0, 0, 1])[:, None]

# generate random weights in the right shape of the NN
ninput = 3
nhidden = 5
nout = 1
nweights = (ninput+1)*nhidden + (nhidden+1) * nout #number of weights
weightlist = [ np.random.rand(nhidden, ninput+1)*2-1, np.random.rand(nout, nhidden+1)*2-1]


#learning rate
alpha = 0.2

# iteration parameters
beta = 400
maxiter = 50
betastep = 700
eqiter = nweights*100 # 2600 if nhidden==5
solutionfound = False

# calculate initial error and set up error storage
error = Error(samples, weightlist, target)
lasterror = error
errorlist = [error]


for b in range(maxiter):
    for n in range(eqiter):
        # select weight randomly by flat index
        index = np.random.randint(nweights)

        # convert flat index to layer index
        if index < (ninput+1)*nhidden:
            nlayer = 0
        else:
            nlayer = 1
            index -= (ninput+1)*nhidden
        
        # get weight indices
        i, j = np.unravel_index(index, weightlist[nlayer].shape)
        # same as
        # i = index // ([ninput, nhidden][nlayer]+1)
        # j = index % ([ninput, nhidden][nlayer]+1)
        #but likely more efficient

        dweight = alpha * (2*np.random.rand(1)-1)

        # get new configuration
        newweights = [weightlist[0].copy(), weightlist[1].copy()]
        newweights[nlayer][i, j] += dweight
        newError = Error(samples, newweights, target)

        # test new configuration
        probability = np.exp(-beta* (newError - error))
        if newError < error or np.random.rand(1) < probability:

            # accept new configuration
            weightlist = newweights
            error = newError

        errorlist.append(error)

        # check if table is memorised
        if np.all(np.heaviside(propagateInput(samples, weightlist)-0.5, 0) == target ):
            print("\nNN output: ", propagateInput(samples, weightlist).reshape(-1))
            print('target values: ', target.reshape(-1))
            print('\nweights: ', *weightlist)

            # abort algorithm
            solutionfound = True
            break
    else:
        print(f"left inner loop with beta={beta} and error {error}")

    if solutionfound:
        break

    beta += betastep
    lasterror = error

if not solutionfound:
    print("no solution was found. Consider changing parameters")

# plot error progress
fig, ax = plt.subplots()
errorlist = np.array(errorlist)
ax.plot(errorlist)
ax.set(xlabel='total iteration number', ylabel='sum of squares error',
        title='Error progress over algorithm runtime')

plt.show()

        
