"""
Jules Faucher
260926201

Phys-512
November 19th 2021

Please see the PDF for the derivations used in this script.
"""

import numpy as np
from matplotlib import pyplot as plt

u = np.linspace(0, 1, 2001)
u = u[1:]

v = -2*u*np.log(u) #See PDF
v_max = v.max() #This max is used to make the rectangular box as tight as possible 

plt.plot(u, v)
plt.xlim(left = 0, right = 1)
plt.ylim(bottom = 0, top = v_max)
plt.title("Ratio of Uniforms", fontsize = 16)
plt.xlabel("$u$", fontsize = 14)
plt.ylabel("$v$", fontsize = 14)
plt.show()

#Here we pick random points in the box and compute their ratios
N = 1000000
u = np.random.rand(N)
v = (np.random.rand(N))*v_max
r = v/u
#We verify which ratios fall inside the curve
accept = u < np.sqrt(np.exp(-r))
efficiency = accept.sum()/N
print(efficiency) #The best efficiency obtained is 67.94%
exp_dist = r[accept] #Keep the accepted ratios

a, b = np.histogram(exp_dist, 100, range = (0, 10))
bb = 0.5*(b[1:] + b[:-1])
pred = np.exp(-bb)*np.sum(accept)*(bb[2] - bb[1])

plt.plot(bb, a, ".", label = "Histogram")
plt.plot(bb, pred, "--", label = "Expectation")
plt.title("Distribution of the Random Numbers Generated", fontsize = 16)
plt.xlabel("$x$", fontsize = 14)
plt.ylabel("$P(x)$", fontsize = 14)
plt.legend()
plt.grid()
plt.show()