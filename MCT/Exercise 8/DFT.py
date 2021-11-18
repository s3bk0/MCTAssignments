import numpy as np
import matplotlib.pyplot as plt

n = 500
xvals = np.linspace(0, 2*np.pi, n)
kays = np.arange(n) - int( n/2 )

# function to be transformed
func = lambda x: np.sin(0.5*x)
yvals = func(xvals) #np.heaviside(xvals-np.pi, 0)

# discrete fourier transform
coeffs = 1/n * np.sum(yvals[:, None] * np.exp(-1j * kays[None, :]\
             * xvals[:, None]), axis=0)

# plot coefficients
plt.figure()
plt.plot(kays, np.abs(coeffs), '-o')

plt.figure()
xvalsplus = np.linspace(-2*np.pi, 4*np.pi, 1000)
backtrafo = np.sum(coeffs[:, None] * np.exp(1j*kays[:, None]*xvalsplus[None, :]), axis=0)
plt.plot(xvalsplus, backtrafo)
plt.plot(xvalsplus, func(xvalsplus))

plt.show()
