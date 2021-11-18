#%%
import numpy as np
import matplotlib.pyplot as plt

n = 100
xvals = np.linspace(0, 2*np.pi, n)
kays = np.arange(n) - int( n/2 )

####################### DFT part ###################################
# function to be transformed
func = lambda x: np.sin(0.5*x)
yvals = func(xvals) #np.heaviside(xvals-np.pi, 0)

# discrete fourier transform
coeffs = 1/n * np.sum(yvals[:, None] * np.exp(-1j * kays[None, :]\
             * xvals[:, None]), axis=0)

# plot coefficients
fregfig, (freqax, fftax) = plt.subplots(2,1, sharex=True)
freqax.plot(kays, np.abs(coeffs), label='amplitudes')
freqax.plot(kays, np.real(coeffs), label="real part")
freqax.plot(kays, np.imag(coeffs), label='imaginary part')
freqax.legend()

xfig, (idftax, ifftax) = plt.subplots(2, 1, sharex= True)

# inverse transformation
xvalsplus = np.linspace(-2*np.pi, 4*np.pi, 1000)
backtrafo = np.sum(coeffs[:, None] * np.exp(1j*kays[:, None]*xvalsplus[None, :]), axis=0)
idftax.plot(xvalsplus, backtrafo)
idftax.plot(xvalsplus, func(xvalsplus))

######################### FFT part ##################################
fftcoeffs = np.fft.fft(yvals, norm='ortho')
sfftcoeffs = np.fft.fftshift(fftcoeffs)
fftax.plot(kays, np.abs(sfftcoeffs), label='amplitudes')
fftax.plot(kays, np.real(sfftcoeffs), label="real part")
fftax.plot(kays, np.imag(sfftcoeffs), label='imaginary part')
fftax.legend()

fftback = np.fft.ifft(fftcoeffs, norm='ortho')

ifftax.plot(xvals, fftback)
#%%
plt.show()
