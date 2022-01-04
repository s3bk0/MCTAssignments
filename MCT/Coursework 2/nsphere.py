import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def inSphere(coords):
    """
    coords : Nsamples x n dimensional array
        coordinates of the drawn samples

    Returns
    -------
    boolean array of length Nsamples indicating position in sphere
    """
    return np.linalg.norm(coords, axis=-1) <= 1

def analyticSphereVol(n):
    return np.pi**(n/2) / gamma(n/2+1)

# number of random points per volume calculation
Nsamples = int(1e6)

# dimensions sphere volumes are calculated for
dimvalues = np.arange(1, 16)
volumes = []

for n in dimvalues:
    # draw Nsamples points in a n dimensional hypercube
    samples = np.random.rand(Nsamples, n)*2-1

    #count values in the sphere and transform this value to a volume
    volume = np.sum(inSphere(samples)) / Nsamples * 2**n
    volumes.append(volume)

fig, ax = plt.subplots()
ax.plot(dimvalues, volumes, label='calculated volumes')
ax.plot(dimvalues, analyticSphereVol(dimvalues), label='analytic volumes')

ax.set(title='volume of a n-dimensional sphere', xlabel='dimension n', ylabel='volume')

ax.legend()
plt.show()
