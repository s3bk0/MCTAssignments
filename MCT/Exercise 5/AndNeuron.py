import numpy as np
import matplotlib.pyplot as plt

def Error(points, weights, target):
    """Error function using square distance from target"""
    return ( Neuron(points, weights) - target )**2

def Neuron(points, weights):
    return np.heaviside( np.sum( points * weights, axis=-1 ), 0 )

def trainNeuron(points, initweights, target, alpha):
    count = 0
    weights = initweights
    # calculate initial total error
    error = np.sum( Error(points, initweights, target) )

    while error != 0:
        for x, t in zip(points, target):
            if Error(x, weights, t)>0:
                weights = weights + ( t - Neuron(x, weights) ) * alpha * x
                print("weights adjusted to w0={:}, w1={:}, w2={:}".format(*weights))
            
        error = np.sum(Error(points, weights, target))
        count += 1
        print(f"Error is now {error}")
    return count, weights



x0 = np.ones(4)
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])

# x_i values stacked along horizontal axis -> analoguous to table
points = np.stack([x0, x1, x2], axis=1)

target = np.array([0, 0, 0, 1])

# learning rates
Nalpha = 15
alphas = np.linspace(0, 1, Nalpha)[1:]

# Number of repititions for a single learning rate
Nstat = 200

iterations = np.zeros((Nstat, Nalpha-1))

for ns in range(Nstat):
    # initialise weights between (-1,1)
    weights = 2 * np.random.rand(3) - 1
    
    for na, alpha in enumerate(alphas):

        iterations[ns, na], _ = trainNeuron(points, weights, target, alpha)

iterations = np.mean(iterations, axis=0)
minind = np.argmin(iterations)

fig, ax = plt.subplots()
ax.plot(alphas, iterations)


print(f"the optimal learning rate is alpha={alphas[minind]}")
plt.show()