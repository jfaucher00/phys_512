"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt

def convo_shift(x, s): #Takes in an array x and shifts it by s.
    
    dirac = np.zeros(len(x)) #Dirac delta function is an array with a single 
    dirac[s] = 1             #non-zero point
    convolution = np.fft.fft(x) * np.fft.fft(dirac)
    return np.fft.ifft(convolution)

Gauss = lambda x, mu, sig : 1/(np.sqrt(2*np.pi) * sig) * np.exp(-(x-mu)**2/sig**2)

x = np.linspace(-10, 10, 100)
y1 = Gauss(x, 0, 1)
y2 = convo_shift(y1, 50)

plt.plot(x, y1, label = "Original Gaussian")
plt.plot(x, y2, label = "Shifted Gaussian")
plt.title("Shifted Gaussian", fontsize = 16)
plt.ylabel("$y$", fontsize = 14)
plt.xlabel("$x$", fontsize = 14)
plt.savefig("./gauss_shift.png", dpi = 400)
plt.legend()
plt.grid()
plt.show()