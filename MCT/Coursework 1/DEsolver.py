import numpy as np
import matplotlib.pyplot as plt
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

x = np.linspace(-20,20,700)
vals = g(x)

plt.plot(x, vals)
plt.plot(x, np.cos(x))


N = 10
dt = 1e-4
t = np.arange(0, (N+1)*dt, dt)

#building problem matrix
plt.figure()
plt.plot(t, g(t))

A = np.diag(g(t)) - np.diag(np.ones(x.shape), 1) - np.diag(np.ones(x.shape), -1)
A[-1,0] = -1
A[0,-1] = -1

slist = linalg.eigvals(A)
print(slist)

plt.show()