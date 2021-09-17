"""
Jules Faucher
260926201

PHYS 512
September 17th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci

dat = np.loadtxt("C:/Users/jules/Desktop/lakeshore.txt") 

def der4(V, dataV, dataT, Inter):   #Evaluates the 4th derivative (used for the error)
    
    if type(V) == float:    #Needed to support both arrays and floats
        V = [V]
    deriv4 = []
    for v in V:
    
        idx = (np.abs(dataV - v)).argmin() #Finds the voltage in the array that 
        Vs = dataV[idx - 2: idx + 3]        #is the closest to the one asked for 
        Ts = dataT[idx - 2: idx + 3]       #Also finds the related temperatures 
        Tmid = Inter(v)
        h = np.amin(Vs[1:]-Vs[:-1])        #H is defined to be the smallest step 
                                            #between data points in the array
        deriv4 += [(Ts[4]-4*Ts[3]+6*Tmid-4*Ts[1]+Ts[0])/h**4] #Adds the forth derivative to a list
    
    return np.array(deriv4)

def lakeshore(V, data, Plot = False):       #Used to find a temperature given a voltage
    Temp, Volt, dVdt = np.transpose(data)
    Volt = np.flip(Volt)
    Temp = np.flip(Temp)
    
    CS = sci.CubicSpline(Volt, Temp)        #Cubic Spline interpolator
    TempInt = CS(V)                         #Interpolated temperature given a voltage
    
    Volt_steps = Volt[1:] - Volt[:-1]       #Finds x-distance between voltages 
    max_step = np.amax(Volt_steps)          #Finds the max step
    error = max_step**4*der4(V, Volt, Temp, CS)/16 #Computes the error
    
    
    if Plot: #Used to plot if required
        x = np.linspace(0.1, 1.6, 2000)
        y = CS(x)
        plt.plot(x, y, '--', label = 'Interpolation')
    
        plt.plot(Volt, Temp, '.', label = "Data Points")
        plt.ylabel("Temperature [K]", fontsize = 14)
        plt.xlabel("Voltage [V]", fontsize = 14)
        plt.title("Voltage as a Function of Temperature", fontsize = 16)
        plt.legend()
        plt.grid()
        plt.show()
    
    return [TempInt, error]

RandomV = np.linspace(0.2, 1.6, 10)
test = lakeshore(RandomV, dat, Plot = True)
print(test)