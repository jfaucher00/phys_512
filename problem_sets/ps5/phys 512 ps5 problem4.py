"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt

def conv_safe(f, g):
    
    lf = len(f)
    lg = len(g)
    
    add_f = max(lf, lg*2-lf)
    add_g = max(lf*2 -lg, lg)
    
    f_extra = np.append(f, np.zeros(add_f))
    g_extra = np.append(g, np.zeros(add_g))
    
    convo = np.fft.fft(f_extra) * np.fft.fft(g_extra)
    convolution = np.fft.ifft(convo)
    convolution = convolution/np.sqrt(np.sum(convolution))
    return convolution[add_f//2: -add_f//2] #Removing the excess from the added zeros

Gauss = lambda x, mu, sig : 1/(np.sqrt(2*np.pi)*sig)*np.exp(-0.5*((x-mu)/sig)**2)

#x = np.arange(-10, 10, 0.1)
x1 = np.arange(-10, 10, 0.1) #Arrays can be different length
x2 = np.arange(-10,  5, 0.1)

mu1 = -1 #Parameters for both Gaussians
mu2 = -5
sig1 = 1
sig2 = 1

mu_true = mu1 + mu2    #Parameters for the true convolution result
sig_true = np.sqrt(sig1**2 + sig2**2)

y1 = Gauss(x1, mu1, sig1)
y2 = Gauss(x2, mu2, sig2)

conv = conv_safe(y1, y2) 

conv_true = Gauss(x1, mu_true, sig_true) #True convolution result for comparison

plt.plot(conv_true, "--", label = "True Conv")
plt.plot(conv, label = "DFT Conv")
plt.plot(y1, label = "$f(x)$")
plt.plot(y2, label = "$g(x)$")
plt.title("Convolution", fontsize = 16)
plt.ylabel("$y$", fontsize = 14)
plt.xlabel("$x$", fontsize = 14)
plt.savefig("./Problem4.png", dpi = 400)
plt.legend()
plt.grid()
plt.show()