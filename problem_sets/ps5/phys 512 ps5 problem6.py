"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize as sci

root = os.path.dirname(__file__)

k_2 = lambda k, C: C/(k**2)
n = 50000

rw = np.cumsum(np.random.randn(n))

x1 = np.linspace(0, np.pi, n//8)
x3 = np.linspace(np.pi, 2*np.pi, n//8)

window1 = 0.5 - 0.5*np.cos(x1)
window2 = np.linspace(1, 1, n - 2*(n//8))
window3 = 0.5 - 0.5*np.cos(x3)
window = np.hstack((window1, window2, window3))
windowed_rw = window*rw

ps = np.abs(np.fft.rfft(windowed_rw))[1:]**2
k = np.arange(1, len(ps)+1, 1)
fit, cov = sci.curve_fit(k_2, k, ps, p0 = [ps[0]])
fit_func = k_2(k, fit[0])
"""
plt.plot(rw)
plt.title("Random Walk", fontsize = 16)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Position", fontsize = 14)
plt.grid()
plt.show()
"""
plt.loglog(ps, label = "Power Spectrum")
plt.loglog(fit_func, label = "C/k^2")
plt.title("Power Spectrum of the Random Walk", fontsize = 16)
plt.xlabel("Time", fontsize = 14)
plt.ylabel("Position", fontsize = 14)
plt.legend()
plt.savefig(root + "/rw_ps.png", dpi = 400)
plt.grid()
plt.show()