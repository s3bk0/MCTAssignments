import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg


def g(x):
    """ function approximating cosine function by straight lines, matching
    it only at the extrema. It is optimised for use with numpy arrays
    """
    # convert x to a numpy array in case it is not. (e.g single value)
    x = np.atleast_1d(x)

    # array containing the final output
    result = np.zeros(x.shape)

    # boolean masks to select values in interval with positive/negative slope
    upmask = -np.sin(x) > 0
    downmask = -np.sin(x) <= 0

    # calculate values
    # in each interval the function is constructed from shifting down a linear 
    # function periodically wich can be achieved by using modulus operations
    # as shown
    result[upmask] =  2* x[upmask]/np.pi % 2 - 1
    result[downmask]= 1 - 2* ( x[downmask]/np.pi % 2 )

    return result

# check correct construction of g by plotting it
x = np.linspace(-20,20,700)
vals = g(x)

plt.plot(x, vals)
plt.plot(x, np.cos(x))


################## Solution of the differential equation #####################
dt = 1e-2 # step size on time scale
N = 2*np.pi / dt
t = np.arange(0, (N+1)*dt, dt) # time values for later reference in the plots

# building the eigenvalue problem:
# diagonal and offdiagonal entries, last parameter specifies 
# offdiagonal position
A = np.diag(2 + 4*dt**2*g(2*t)) \
    - np.diag(np.ones(t.shape[0]-1), 1) \
    - np.diag(np.ones(t.shape[0]-1), -1)

# matrix elements that ensure periodic boundary conditions
A[-1,0] = -1
A[0,-1] = -1



# solving the eigenvector problem
slist, solutions = linalg.eig(A)

# getting the indices of the lowest eigenvalues
mininds = (slist).argsort()
# print(mininds, slist[mininds])

# set up the plot
fig, axes = plt.subplots(5,1, sharex=True, figsize=(11,7))

# fill subplots with solutions from the eigenvalue problem
for i, ax in enumerate(axes):
    ax.set(ylabel="w(t)", title=f"Eigenvalue s={round(slist[mininds[i]] / dt**2, 3)}")
    ax.plot(t, solutions[:,mininds[i]], color=f"C{i}")
    ax.grid()
axes[-1].set(xlabel="time t")

plt.tight_layout()
plt.show()