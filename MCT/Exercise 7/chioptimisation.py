import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def linear(x, m, b):
    return m*x+b

def chisq(linparams, xdata, ydata, sig):
    """chi square of a linear model
    linparams : array of lenght 2 with linear parameters m and b
    """
    m, b = linparams
    if type(m)==np.ndarray:
        m = m[..., None]
        b = b[..., None]
    return np.sum( (linear(xdata, m, b) - ydata)**2 / sig**2)

def reldiff(xold, xnew):
    return np.abs( ( xold-xnew )/ xnew)

def gradient(func, x0, funcargs=(), dx=0.001):
    x0 = np.atleast_1d(x0)
    N = x0.size

    f0 = func(x0, *funcargs)
    Xph = ( x0 * np.ones((N,N))).T + dx*np.eye(N)
    return (func(Xph, *funcargs) - f0) / dx

def GradientDescent(func, x0, funcargs=(), rate=1e-2, maxiter=200, relaccuracy=1e-7):
    xold = x0
    for k in range(maxiter):
        grad = gradient(func, xold, funcargs, dx=1e-7)
        grad /= np.linalg.norm(grad)
        xnew = xold - rate * grad
        err = np.sum(reldiff(xold, xnew))

        print(k, xnew, err, func(xnew, *funcargs))
        if err < relaccuracy:
            print(f"exiting: xold={xold}, xnew={xnew}, err={err}")
            break
        xold = np.copy(xnew)
    return xnew

# sig = 0.2*np.ones(5)
# x = np.arange(5)
# y = np.array([1.0, 2.8, 2.8, 2.0, 4.9])
N = 20
x = np.linspace(0, 50, N)
y = linear(x, 2, 0.5) + 2*np.random.rand(N) - 1.0
sig = np.ones(N) * 2 / np.sqrt(12)

initparams =np.array([0.7, 2.8])
params = GradientDescent(chisq, initparams, (x, y, sig), rate=10e-3, maxiter=300 )
print(initparams)
print("results self written algorithm:", params,
             f"chi={chisq(params, *(x, y, sig))}")

cfparams, _ = curve_fit(linear, x, y, initparams, sigma=sig, absolute_sigma=True)
print("results curvefit:", cfparams,
             f"chi={chisq(cfparams, *(x, y, sig))}")

fig, (ax1, ax2) = plt.subplots(2,1, sharex='col', figsize=(9, 7), \
          gridspec_kw={'height_ratios': [2,1]})

xvals = np.linspace(x[0], x[-1], 100)
ax1.errorbar(x, y, yerr=sig, capsize=2, fmt='o')
ax1.plot(xvals, linear(xvals, *params))
ax1.plot(xvals, linear(xvals, *cfparams))

ax2.errorbar(x, linear(x, *params)-y, yerr=sig, fmt='o', capsize=2, label='self')
ax2.errorbar(x, linear(x, *cfparams)-y, yerr=sig, fmt='o', capsize=2, label='curve fit')
ax2.legend()
ax2.grid()
ax1.grid()

plt.show()






