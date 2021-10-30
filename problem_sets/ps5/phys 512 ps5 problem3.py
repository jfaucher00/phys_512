"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt

def convo_shift(x, s): #Takes in an array x and shifts it by s.
    
    dirac = np.zeros(len(x))
    dirac[s] = 1
    convolution = np.fft.fft(x) * np.fft.fft(dirac)
    return np.fft.ifft(convolution)

def corr(y1, y2):
    correlation = np.fft.ifft(np.fft.fft(y1) * np.conj(np.fft.fft(y2)))
    return correlation/np.sqrt(np.sum(correlation))

Gauss = lambda x, mu, sig : 1/(np.sqrt(2*np.pi)*sig)*np.exp(-((x-mu)/sig)**2)

def shift_corr(y1, s):
    y2 = convo_shift(y1, s)
    return corr(y1, y2), y2

x = np.linspace(-20, 20, 400)
y = Gauss(x, -15, 1)

s = 100

dirac = np.zeros(len(x))
dirac[s] = 1

result, y2 = shift_corr(y, s)

plt.plot(x, result, label = "Correlation")
plt.plot(x, y, label = "Original Gauss.")
plt.plot(x, y2, label = "Shifted Gauss.")
plt.title("Correlation of Shifted Gaussians", fontsize = 16)
plt.ylabel("$y$", fontsize = 14)
plt.xlabel("$x$", fontsize = 14)
plt.savefig("Problem3.png", dpi = 400)
plt.grid()
plt.legend()
plt.show()