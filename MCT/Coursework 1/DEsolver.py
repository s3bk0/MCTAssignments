import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import numpy.linalg as linalg


def g(x):
    x = np.atleast_1d(x)
    result = np.zeros(x.shape)

    #select values in interval with positive/negative slope
    upmask = -np.sin(x) > 0
    downmask = -np.sin(x) <= 0

    #calculate values
    result[upmask] =  2* x[upmask]/np.pi % 2 - 1
    result[downmask]= 1 - 2* ( x[downmask]/np.pi % 2 )

    return result

# x = np.linspace(-20,20,700)
# vals = g(x)

# plt.plot(x, vals)
# plt.plot(x, np.cos(x))


dt = 1e-2
N = 2*np.pi / dt
t = np.arange(0, (N+1)*dt, dt)

# building problem matrix
# plt.figure()
# plt.plot(t, g(t))

A = np.diag(2 + 4*dt**2*g(2*t)) - np.diag(np.ones(t.shape[0]-1), 1) - np.diag(np.ones(t.shape[0]-1), -1)
A[-1,0] = -1
A[0,-1] = -1

slist, solutions = linalg.eig(A)
print(slist)

# lowest indices
mininds = slist.argsort()
print(mininds, slist[mininds])

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