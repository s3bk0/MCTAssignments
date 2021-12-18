import numpy as np
import matplotlib.pyplot as plt

def backtrafo(x, a0, am, ak, bk):
    k = np.arange(1, 8)[:, None]
    return (a0/2 + am/2 * np.cos(8*x) 
                    + np.sum(ak[:,None]*np.cos(k*x) 
                    + bk[:,None]*np.sin(k*x), axis=0)) / 16

xvals = np.arange(0, 2*np.pi, np.pi/8)
kindex = np.arange(9)
data = np.array([-0.2, -0.1, 0.3, 0.2, 
                0.4 , 0.5, 0, -0.4, 
                -0.4, -0.2, 0.1, 0.2, 
                0.2, 0.1, 0.1, -0.1])
# data = np.cos(3*xvals)+ np.sin(4*xvals)

fftcoeffs = np.fft.fft(data, norm='backward')

# calculate the coefficients of the real valued FT
# am and a0 are already real but throw a warning if not made real
am = 2*np.real(fftcoeffs[8]) 
a0 = 2*np.real(fftcoeffs[0])
ak = 2*fftcoeffs[1:8].real
bk = 2*-fftcoeffs[1:8].imag 

trigparams = [a0, am, ak, bk]

print('The fourier coefficients are')
print('a0={:.3f}, a1={:.3f}, a3={:.3f}, a4={:.3f}, a5={:.3f}, a6={:.3f}, a7={:.3f}, a8={:.3f}'.format(a0, *ak, am))
print('b1={:.3f}, b3={:.3f}, b4={:.3f}, b5={:.3f}, b6={:.3f}, b7={:.3f}'.format(*bk))
                

fig, (ax1, ax2) = plt.subplots(2,1)
plt.subplots_adjust(hspace=0.4)
ax1.set(title='fft interpolation of a data set')
ax1.plot(xvals, data, 'o', label='given samples')

ax2.plot([0], [a0],'o', color='C0')
ax2.plot(kindex[1:8], ak, 'o', label=r'$a_k$')
ax2.plot(kindex[1:8], bk, 'o', label=r'$b_k$')
ax2.plot([8], [am],'o', color='C0')

ax2.set(title='real value fourier coefficients', ylabel='coefficent value',
        xlabel='index k')
ax2.legend()

# fftback = np.fft.ifft(fftcoeffs, norm='backward')
xcont = np.linspace(0, 2*np.pi, 100)
ax1.plot(xvals, backtrafo(xvals, *trigparams), 'x', color='red', label='inverse FFT points')
ax1.plot(xcont, backtrafo(xcont, *trigparams), label='inverse FFT function')
ax1.legend()

plt.show()