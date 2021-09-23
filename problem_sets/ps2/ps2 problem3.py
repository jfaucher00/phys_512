"""
Jules Faucher
260926201

PHYS 512
September 24th, 2021
"""

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
import matplotlib.pyplot as plt

x = np.linspace(0.5, 1.0, 10) #Points used to model log2
x_str = 4*x - 3 #Mapping these points from [0.5, 1.0] to [-1.0, 1.0]
y = np.log2(x)
poly = chebfit(x_str, y, 7) #Computing the polynomials to use

def mylog2(x, polynomials):
    
    e = np.exp(1)  #We need the log2(e) to change the basis of our logarithm
    mant_e, exp_e = np.frexp(e) #The following lines follow the logic presented in the PDF
    y_e = chebval(4*mant_e - 3, polynomials)
    log_e = exp_e + y_e
    
    mant, exp = np.frexp(x) #We repeat these steps but for the function's input
    y = chebval(4*mant - 3, polynomials)
    
    return (exp + y)/log_e

#These following lines can be used for testing
x = np.linspace(0.1, 10)
plt.plot(x, np.log(x))
plt.plot(x, mylog2(x, poly))
#plt.plot(x, np.abs(np.log(x) - mylog2(x, poly)))


#These lines were used to determine the best number of polynomials to use
if False:
    x = np.linspace(0.5, 1.0, 10)
    y = np.log2(x)
    x_str = 4*x - 3 #Values of x mapped from [0.5, 1.0] to [-1.0, 1.0]

    xx = np.linspace(0.5, 1.0) #The following values are used to plot results
    yy = np.log2(xx)
    xx_str = 4*xx - 3  #Mapping xx from [0.5, 1.0] to [-1.0, 1.0]

    poly = chebfit(x_str, y, 7)
    test2 = chebval(xx_str, poly)

    plt.plot(x, y, '.')
    plt.plot(xx, test2)
    plt.show()
    plt.plot(xx, np.abs(test2-yy))