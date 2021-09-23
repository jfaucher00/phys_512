"""
Jules Faucher
260926201

PHYS 512
September 24th, 2021

Note: This is code from class with a couple of lines added to "cache" values
of f(x) as they are computed for later use.
"""

import numpy as np

call_count = 0 #We use this variable to track the number of calls to f(x) needed.

def integrate_adaptive(fun, x0, x1, tol, extra = None):
    
    global call_count
    
    x = np.linspace(x0,x1,5)
    
    if extra is None:
        y = fun(x)
        call_count += 5
        extra = np.array([x, y]) #This array will store known points of f(x)
    else:
        y = []
        for i in x:
            if i not in extra[0]: #We will stack new known points as we go
                temp = np.array([[i],[fun(i)]]) #This is a new column with x and f(x)
                call_count += 1
                extra = np.hstack((extra, temp)) #We add this column to extra
            
            idx = int(np.where(extra[0] == i)[0][0]) #This part acts as a python dictionnary
            y += [ extra[1][idx] ]      #We could not use a dict since floats can't be indexes
        
    
    dx=(x1-x0)/(len(x)-1)
    area1=2*dx*(y[0]+4*y[2]+y[4])/3 #coarse step
    area2=dx*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3 #finer step
    err=np.abs(area1-area2)
    
    if err<tol:
        print( len(extra[0]) )
        return area2
    else:
        xmid=(x0+x1)/2
        left=integrate_adaptive(fun,x0,xmid,tol/2)
        right=integrate_adaptive(fun,xmid,x1,tol/2)
        return left+right
    
#Test cases
x0=0
x1=1

if True:
    ans = integrate_adaptive(np.exp,x0,x1,1e-6)
    print(ans-(np.exp(x1)-np.exp(x0)))
    """
    Without the extra parameter, there are 75 calls to the function.
    With it, we reduce the number of calls to 33.
    """
    
    
else:
    lorentz = lambda x : 1/(1+x**2)
    ans = integrate_adaptive(lorentz,x0,x1,1e-7)
    print(ans-(np.arctan(x1)-np.arctan(x0)))
    """
    Without the extra parameter, there are 165 calls to the function.
    With it, we reduce the number of calls to 69.
    """