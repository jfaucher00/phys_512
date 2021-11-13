"""
Jules Faucher
260926201

Phys-512
November 19th 2021

The main strategy was to loop through multiple values of "a" and look at
the resulting plots. I concluded that at 1.100 < a< 1.107, we can clearly
see lines in the 2D plot (See problem1_C-RNG.png).

This behaviour was not seen when using the random number generator from
numpy. I have included an example (See problem1_numpy-RNG.png).

I could not get the last part to work on Windows.
"""

import numpy as np
import matplotlib.pyplot as plt

x, y, z = np.loadtxt("./rand_points.txt").T
lenght = len(x)

"""
for a in np.arange(1.09, 1.114, 0.001):
        
    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.0, 2.0, 2.0, 2.0))

    xs = a*x + y
    plt.plot(xs, z, ".")
    plt.title("a = " + str(round(a, 4)))
    plt.show()
"""

a = 1.04   #Example with a = 1.04, where we can clearly see lines.
  
fig1 = plt.figure(1)
frame1 = fig1.add_axes((0.0, 4.0, 4.0, 4.0))

xs = a*x + y
plt.plot(xs, z, ".")
plt.title("a = " + str(round(a, 4)), fontsize = 30)
plt.xlabel("$a x + y$", fontsize = 24)
plt.ylabel("$z$", fontsize = 24)
plt.show()

#Now with numpy. I reuse the same strategy as before.

x = np.random.rand(lenght)
y = np.random.rand(lenght)
z = np.random.rand(lenght)

for a in np.arange(0.1, 3, 0.05):
        
    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((0.0, 2.0, 2.0, 2.0))

    xs = a*x + y
    plt.plot(xs, z, ".")
    plt.title("a = " + str(round(a, 4)))
    plt.show()
#I could not see any structures in any plots generated.

