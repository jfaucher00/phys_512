"""
Jules Faucher
260926201

PHYS 512
September 17th, 2021
"""

import numpy as np

def deriv(f, x, dx): #Same formula as derived on paper
    return (f(x-2*dx)+8*f(x+dx)-8*f(x-dx)-f(x+2*dx))/(12*dx)

x = np.linspace(-1, 1, 10)         #Some x values to try the derivatives on
dx = 1.6e-06                       #Step size estimated for exp(x)
exp0_01 = lambda x: np.exp(0.01*x) #Defining exp(0.01*x)

y1 = deriv(np.exp, x, dx)          #Derivatives of exp(x)
y01 = np.exp(x)                    #Exact values for the derivative 
frac1 = y1/y01-1                   #Fractional error 

y2 = deriv(exp0_01, x, dx/100)     #Derivatives of exp(0.01*x) 
y02 = 0.01*exp0_01(x)              #Exact values for the derivative 
frac2 = y2/y02-1                   #Fractional error 

print(frac1, "\n", frac2)