import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as spi
"""
Jules Faucher
260926201

PHYS 512
September 17th, 2021
"""

def rateval(x,p,q): #Function to evalute rational functions from class
    top=0
    for i,par in enumerate(p):
        top=top+par*x**i
    bot=1
    for i,par in enumerate(q):
        bot=bot+par*x**(i+1)
    return top/bot

def ratfit(y,x,n,m, Pinv = False): #Function to fit rational function from class
    npt=len(x)                              
    assert(len(y)==npt)
    assert(n>=0)
    assert(m>=0)
    assert(n+1+m==npt)

    top_mat=np.empty([npt,n+1])
    bot_mat=np.empty([npt,m])
    for i in range(n+1):
        top_mat[:,i]=x**i
    for i in range(m):
        bot_mat[:,i]=y*x**(i+1)    
    mat=np.hstack([top_mat,-bot_mat])
    
    if Pinv == True:
        pars=np.linalg.pinv(mat)@y
    else:
        pars=np.linalg.inv(mat)@y
        
    p=pars[:n+1]
    q=pars[n+1:]
    return p,q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

def plotter(x, fun, function_name): #Function that plots the differences between
                                    #a function and its interpolations
    xx = np.linspace(x[0], x[-1], 100) #array of x for ploting
    y = fun(x)  #Array of points to fit
    yy = fun(xx)    #Array of points used to finod the difference
       
    PI = spi.KroghInterpolator(x, y) #Polynomial interpolator
    CS = spi.CubicSpline(x, y)       #Cubic Spline interpolator
    m=len(y)//2
    n=len(y)-m-1
    p, q = ratfit(y, x, n, m)        #Rational function interpolation
    r, s = ratfit(y, x, n, m, Pinv = True) #Rational function interpolation with Pinv

    y_dict = {"Polynomial": PI(xx), "Cubic Spline": CS(xx), #Storing all results 
              "Rational": rateval(xx, p, q), "Rational (Pinv)": rateval(xx, r, s)}
    
    for i in y_dict:        #Plots all differences between an interpolation and the actual function
        values = y_dict[i]
        plt.title((i + " Interpolation of " + function_name), fontsize = 16)
        plt.axhline(color = "black")
        plt.plot(xx, values-yy)
        plt.grid()
        plt.show()

x = np.linspace(-np.pi/2, np.pi/2, 7)  #Couple of points interpolated
plotter(x, np.cos, "cos(x)")           #Plotting 

x = np.linspace(-1, 1, 7)              #Couple of points to be interpolated 
lorentz = lambda x: 1/(1 + x**2)       #Definition of a Lorentzian 
plotter(x, lorentz, "Lorentzian")      #Plotting 