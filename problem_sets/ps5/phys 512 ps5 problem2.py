"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt


def corr(y1, y2):
    correlation = np.fft.ifft(np.fft.fft(y1) * np.conj(np.fft.fft(y2)))
    return correlation/np.sqrt(np.sum(correlation))


Gauss = lambda x, mu, sig : 1/(np.sqrt(2*np.pi)*sig) *np.exp(-((x-mu)/sig)**2)

x = np.linspace(-10, 10, 100)
y = Gauss(x, 2, 1)

correlation = corr(y, y)

plt.plot(x, correlation, label = "Correlation")
plt.plot(x, y, label = "Gaussian")
plt.xlabel("$x$", fontsize = 14)
plt.ylabel("$y$", fontsize = 14)
plt.title("Correlation of a Gaussian with Itself", fontsize = 16)
plt.savefig("./correlation1.png", dpi = 400)
plt.legend()
plt.grid()
plt.show()