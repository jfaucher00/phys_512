"""
Jules Faucher
260926201

PHYS 512
September 17th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt

def ndiff(fun, x, full = False):
    
    eps_m = 2**(-52)
    delta = 0.001      #Small delta to evaluate the third derivative
        
    d3 = (fun(x + 2*delta) - fun(x - 2*delta) + 2*fun(x - delta) - 2*fun(x + delta))/(2*delta**3)
    d3_f = np.where(d3<eps_m, eps_m**(1/3), d3) #Swaps value of the 3rd deriv. by the machine's precision when the fromer is too small to avoid divisions by zero
    dx = np.cbrt(eps_m*fun(x)/d3_f)     #Computes the appropriate step size.
    deriv = (fun(x + dx) - fun(x - dx))/(2*dx) #Calculates the derivative's value
    
    error = np.abs(np.cbrt(eps_m**2*fun(x)**2*d3_f)) #From the textbook
    
    if full:
        return [deriv, error]
    else:
        return deriv

#The following is used for testing
x = np.linspace(0, 1, 1000)
y = lambda x: np.exp(x)
z = lambda x: np.exp(x)

DyDx = ndiff(y, x, full = True)

diff = np.abs(DyDx[0] - z(x))

#plt.plot(x, DyDx[0])
#plt.plot(x, z(x))
plt.plot(x, diff, label = "Difference (exp(x)-DyDx)")
plt.plot(x, DyDx[1], label = "Computed Error")
plt.legend()