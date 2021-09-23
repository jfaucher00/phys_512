"""
Jules Faucher
260926201

PHYS 512
September 24th, 2021
"""

import scipy.integrate as sc
import numpy as np
import matplotlib.pyplot as plt

def integrate(f, a, b, N = 99):     #Generic Simpson Integrator
    dx = (b-a)/N 
    x = np.linspace(a, b, N)
    area = dx * (f(x) + 4*f(x + dx/2) + f(x + dx))/6
    return np.sum(area), 0

def dE(theta, z, eps, sigma, R):    #Definition of the integrand
    dE1 = (sigma * R**2 * np.sin(theta)) * (z - R*np.cos(theta))
    dE2 = 2*eps*np.sqrt(R**2 + z**2 - 2*R*z*np.cos(theta))**(3)
    return dE1/dE2
    
def alongZ(Z, eps, sigma, R, integrator):   #Integrates over theta at multiple values of z
    integral = []       #This list will accumulate the integrals at different z values
    for z in Z:
        reduced_dE = lambda t : dE(t, z, eps, sigma, R) #Rewriting dE so it has only one variable
        integral += [integrator(reduced_dE, 0, np.pi)]
    return np.array(integral)

def plotter(integrator, z1, z2, eps = 1, sigma = 1, R = 1): #Rearanges the data to plot it
    Z = np.linspace(z1, z2, 501)
    E, error = np.transpose(alongZ(Z, eps, sigma, R, integrator))
    plt.plot(Z, E)
    
plotter(sc.quad, 0, 10)
plotter(integrate, 0, 10)
plt.title("Electric Field as a Function of Distance", fontsize = 16)
plt.xlabel("Distance from the Center of the Sphere", fontsize = 14)
plt.ylabel("Electric Field Magnetude", fontsize = 14)
plt.grid()
plt.show()