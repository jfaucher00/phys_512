"""
Jules Faucher
260926201

Phys-512
November 19th 2021
"""

import numpy as np
import matplotlib.pyplot as plt

#First, we need to figure out which bounding function is better.
#i.e. the one that fits the closest to our distribution and that 
#always yield a greater value.

Exp = lambda x : np.exp(-x)
Lorentz = lambda x : 1/(1+x**2)
Gaussian = lambda x : np.exp(-(x**2)/2)/np.sqrt(2* np.pi)
Power = lambda x : x**(-2)

functions_law = {"Exponential": Exp, 
                 "Lorentzian": Lorentz, 
                 "Gaussian": Gaussian, 
                 "Power Law": Power}

#I excluded the power law since it diverged at x = 0, and ruined the plot.

functions = {"Exponential": Exp, 
             "Lorentzian": Lorentz, 
             "Gaussian": Gaussian}

def comparison(x1, x2, funcs):
    
    x = np.linspace(x1, x2)

    for f_name, f in funcs.items():
        plt.plot(x, f(x), label = f_name)
    
    plt.title("Comparison of Different Bounding Functions", fontsize = 16)
    plt.xlabel("$x$", fontsize = 14)
    plt.xlabel("$y", fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()
    
comparison(0, 2, functions)
#comparison(20, 26, functions)

#We can conclude that the lorentzian function is better since the Gaussian
#function yields smaller values than the exponential. On the other hand, the
#power law yields larger values than the lorentzian, making it inefficient.


def lorentzians(n): #Creates lorentz distributed values.
    q = np.pi*(np.random.rand(n)-0.5)
    return np.tan(q)

n = 10000000
t = lorentzians(n)
y = np.random.rand(n)/(1+t**2) #Random height

bins = np.linspace(0,10,501)

cents = 0.5*(bins[1:]+bins[:-1])
pred = 1/(1+cents**2)  #Expected values
pred = pred/pred.sum() #Normalization

accept = y < Exp(t) #Verifying if the values are inside the exp distribution
efficiency = accept.sum()/n #Fraction of points accepted
print(efficiency) #The best efficiency obtained is 81.83%
t_use = t[accept]

aa, bb = np.histogram(t_use, bins)
aa = aa/aa.sum()
pred = Exp(cents)
pred = pred/pred.sum()
plt.plot(cents, aa, '*', label = "Histogram")
plt.plot(cents, pred, '--', label = "Expectation")
plt.title("Distribution of the Random Numbers Generated", fontsize = 16)
plt.ylabel("$P(x)$", fontsize = 14)
plt.xlabel("$x$", fontsize = 14)
plt.legend()
plt.grid()
plt.show()