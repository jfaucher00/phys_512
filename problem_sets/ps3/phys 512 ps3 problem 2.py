"""
Jules Faucher
260926201

Phys 512
October 8th, 2021
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

years = 1 #Definition of certain units
days = years/365.25
hours = days/24
minutes = hours/60
seconds = minutes/60

#Here, we can adjust the starting quantities of each element in the decay chain.
N0 = {"U238" : 1, "Th234": 0, "Pa234": 0, "U234" : 0, "Th230": 0,
      "Ra226": 0, "Rn222": 0, "Po218": 0, "Pb214": 0, "Bi214": 0,
      "Po214": 0, "Pb210": 0, "Bi210": 0, "Po210": 0, "Pb206": 0}

#Here, we can change the values of the half-lives.
HL = {"U238" : 4.468e9*years, "Th234": 24.10*days, "Pa234": 6.70*hours,
      "U234" : 2.455e5*years, "Th230": 7.538e4*years,"Ra226": 1600.0*years,
      "Rn222": 3.8235*days, "Po218": 3.10*minutes, "Pb214": 26.8*minutes,
      "Bi214": 19.9*minutes, "Po214": 1.643e-4*seconds, "Pb210": 22.3*years,
      "Bi210": 5.015*years, "Po210": 138.376*days}


def fun(t, N): #This sets up the stiff ODE following the same logic as in class.
    T = HLv
    dNdt = np.zeros(len(T)+1)
    dNdt[0] = -N[0]/T[0]
    
    for i in range(1, len(T)): 
        dNdt[i] = N[i-1]/T[i-1] - N[i]/T[i]
    
    dNdt[len(T)] = N[len(T)-1]/T[len(T)-1]
    
    return dNdt*np.log(2)

if False: #Toggle to see part a)
    y0 = np.asarray(list(N0.values()))  #Calling all initial quantities
    HLv = np.asarray(list(HL.values())) #Getting all half-lives
    x0 = 0 #Initial time
    x1 = 5*HL["U238"] #Final time
    decay = integrate.solve_ivp(fun, [x0, x1], y0, method = "Radau")

    y = decay.y
    t = decay.t
    
    for i in y: #Loop to plot the quantities of every element's decay
        plt.plot(t, i)
    plt.title("Decay Chain of U-238", fontsize = 16)
    plt.xlabel("Time Elapsed [Years]", fontsize = 14)
    plt.ylabel("Relative Quantities [Total = 1]", fontsize = 14)
    plt.grid()
    plt.show()

#---------- Part B ----------#

def fun_b(t, N): #Same as the fun function but only for 2 elements
    T = HL1
    dNdt = np.zeros(2)
    dNdt[0] = -N[0]/T
    dNdt[1] = N[0]/T
    return dNdt*np.log(2)

if False:
    y0 = [1, 0] #Initial quantities
    HL1 = 4.468e9*years #Half-life of U238
    x0 = 0 #Initial time
    x1 = 2*HL1 #Final time
    decay = integrate.solve_ivp(fun_b, [x0, x1], y0, method = "Radau")
    
    y_U, y_Pb = decay.y
    t = decay.t
    
    plt.plot(t, y_Pb/y_U, label = "Ratio")
    plt.axvline(HL1, color = "black", label = "Half-Life of U238")
    plt.title("Pb210 to U238 Ratio", fontsize = 16)
    plt.xlabel("Time Elapsed [Years]", fontsize = 14)
    plt.ylabel("Pb210/U238", fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()

if True:
    y0 = [1, 0, 0]  #Initial quantities
    HLv = np.array([HL["U234"], HL["Th230"]]) #Calling needed half-lives
    x0 = 0
    x1 = 60*HLv[0]
    decay = integrate.solve_ivp(fun, [x0, x1], y0, method = "Radau")
    
    y_U, y_Th, y_Pb = decay.y
    t = decay.t
    plt.plot(t, y_Th/y_U)
    plt.title("Th230 to U234 Ratio", fontsize = 16)
    plt.xlabel("Time Elapsed [Years]", fontsize = 14)
    plt.ylabel("Th230/U234", fontsize = 14)
    plt.grid()
    plt.show()
    plt.show()