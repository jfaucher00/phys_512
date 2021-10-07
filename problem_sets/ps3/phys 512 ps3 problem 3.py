"""
Jules Faucher
260926201

Phys 512
October 8th, 2021
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

path = "./dish_zenith.txt"
x, y, z = np.transpose(np.loadtxt(path))

if False: #Toggle to see the parabola
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, ".")
    plt.show()

paraboloid = lambda r, x0, y0, z0, a: a*((r[0] - x0)**2 + (r[1] - y0)**2) + z0 #Might be useful

A = np.array([x**2 + y**2, x, y, np.ones(len(x))])  #Defining the A matrix

U, S, V = np.linalg.svd(A)  #Using SVD instead of inversion
Sinv_1 = np.diag(S**(-1))
Sinv = np.hstack((Sinv_1, np.zeros((4,471)))).T #Building S^(-1) as it should have been

m = ((V.T)@Sinv@(U.T)).T@(z) #Computing the best-fit parameters

#------------ Part B ------------

a, B, C, D = m

x0 = -B/2/a
y0 = -C/2/a
z0 = D - a*x0**2 - a*y0**2

#------------ Part C ------------

f = 0.25/a

errors = paraboloid([x, y], x0, y0, z0, a) - z
mean = np.mean(errors)
std  =np.std(errors) #Our error on z if the noise is gaussian

if True: #Toggle to see 2D residuals
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, errors, ".")
    plt.title("Residuals along the x-Axis", fontsize = 16)
    plt.ylabel("y [mm]", fontsize = 14)
    plt.xlabel("x [mm]", fontsize = 14)
    plt.show()
    
    plt.plot(x, errors, "o")
    plt.axhline()
    plt.title("Residuals along the x-Axis", fontsize = 16)
    plt.ylabel("Residuals", fontsize = 14)
    plt.xlabel("x [mm]", fontsize = 14)
    plt.grid()
    plt.show()
    
    plt.plot(y, errors, "o")
    plt.axhline()
    plt.title("Residuals along the y-Axis", fontsize = 16)
    plt.ylabel("Residuals", fontsize = 14)
    plt.xlabel("y [mm]", fontsize = 14)
    plt.grid()
    plt.show()

gaussian = lambda x, mu, sigma, A: A*np.exp(-((x-mu)/sigma)**2)
x = np.linspace(-10, 15)

plt.hist(errors, label = "Noise Distribution") #Verifying if the noise is Gaussian
plt.plot(x, gaussian(x, mean, std, 140), label = "Gaussian") #We plot a Gaussian with the same mean and variance
plt.title("Distribution of the Noise", fontsize = 16)
plt.ylabel("Counts", fontsize = 14)
plt.xlabel("Error [mm]", fontsize = 14)
plt.legend()
plt.show()


N = std**2*np.eye(475)
cov1 = np.linalg.pinv(A@np.linalg.inv(N)@A.T) #Computing the covariance matrix

error_a = cov1[0][0]**(1/2)
error_f = f*error_a/a

print("The focal lenght is",f, "+/-", error_f, "mm")