"""
Jules Faucher
260926201

Phys 512
October 8th, 2021
"""

import numpy as np
from matplotlib import pyplot as plt


def func(x, y): #Function to be tested
    return y/(1 + x**2)

def rk4_step(fun,x,y,h, k1=None, k1_out = False): #rk4 step as seen in class
    if k1 is None:
        k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    
    if k1_out:
        return y+dy, k1
    else:
        return y+dy

def rk4_stepd(fun,x,y,h):   #rk4 with fifth order corrections, as the derived on the pdf
    y1, funeval = rk4_step(fun, x, y, h, k1_out = True)
    y2_1 = rk4_step(fun, x, y, h/2, k1=funeval/2)
    y2 = rk4_step(fun, x+h/2, y2_1, h/2)
    delta = y2 - y1
    return y2 + delta/15

def rk4_integrator(function, x_i, x_f, steps, y_x_i, technique = rk4_step): #Function to make the use of the previous functions more convenient
    x = np.linspace(x_i, x_f, steps + 1)
    h = (x_f - x_i)/(steps)
    y = np.zeros(steps + 1)
    y[0] = y_x_i
    for i in range(steps):
        y[i+1] = technique(function,x[i],y[i], h)
    return x, y


x, y = rk4_integrator(func, -20, 20, 200, 1)
plt.plot(x, y, label = "Numerical Soln")
soln = np.exp(-np.arctan(-20))*np.exp(np.arctan(x))
plt.plot(x, soln, "--", label = "Solution")
error = "Max error is " + str(round(np.amax(np.abs(soln - y)), 5))
plt.text(-20.5, 17, error)
plt.title("Without Fifth Order Corrections", fontsize = 16)
plt.xlabel("x", fontsize = 14)
plt.ylabel("y", fontsize = 14)
plt.legend()
plt.show()

x, y = rk4_integrator(func, -20, 20, 67, 1, rk4_stepd)
plt.plot(x,y, label = "Numerical Soln")
soln = np.exp(-np.arctan(-20))*np.exp(np.arctan(x))
plt.plot(x, soln, "--", label = "Solution")
error = "Max error is " + str(round(np.amax(np.abs(soln - y)), 5))
plt.text(-20.5, 17, error)
plt.title("With Fifth Order Corrections", fontsize = 16)
plt.xlabel("x", fontsize = 14)
plt.ylabel("y", fontsize = 14)
plt.legend()
plt.show()