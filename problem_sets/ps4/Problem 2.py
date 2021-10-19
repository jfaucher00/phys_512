"""
Jules Faucher
260926201

Phys 512
October 15th, 2021
"""

import numpy as np
import camb
from matplotlib import pyplot as plt
import time
import os

dirname = os.path.dirname(__file__)

def get_spectrum(pars,lmax=3000): #Same function as the original
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]
    return tt[2:]


def Grad(fun, x):   #Computes the gradient of a function
    dim = len(x)
    h = 1e-8 
    hI = h*np.eye(dim)*pars #I am using the guess parameters to scale the step sizes 
    copies = np.outer(x, np.ones(dim)).T #Creates a matrix where every row are x
    xplush = copies + hI #Adds a step to every elements on the diagonal
    xminush = copies - hI #Substracts a step on the diagonal
    grad = [] #Accumulates the results
    for i in range(dim):
        grad += [(fun(xplush[i]) - fun(xminush[i]))/2/h/pars[i]] #Differentiation
    return np.array(grad)


pars = np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])

def fit_newton(m, fun, y, errs, max_iter = 10, chi_tol = 0.02):
    
    len_y = len(y) 
    current_iter = 0 #Tracks how many iterations were done
    chi1 = 4*len_y #Chi1 and 2 are initiated as very large and very different values
    chi2 = 3*len_y
    
    Ninv = np.eye(len_y)/(errs**2) #Inverse noise matrix
    
    while current_iter < max_iter and (chi2 + chi_tol < chi1):
        
        f = fun(m)[0:len_y]      # model
        r = y - f                # residuals
        pre_df = Grad(fun, m)    # derivative of f
        df = pre_df.T[0:len_y].T # lenght of df is adjusted to fit the data
        
        lhs = df@Ninv@df.T       
        rhs = df@Ninv@r.T
        cov = np.linalg.inv(lhs) #Covariance matrix
        dm = cov@rhs
        m += dm
        
        chi1 = chi2                 #Updating both chi1 and chi2
        chi2 = np.sum( (r/errs)**2)
        current_iter += 1
        print("iter:", current_iter)
        
    pars_errs = np.sqrt(np.diag(cov))    
    return m, pars_errs, cov

planck = np.loadtxt(dirname + '/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec = planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

fit_pars, fit_errs, cov = fit_newton(pars, get_spectrum, spec, errs)
model = get_spectrum(fit_pars)[0:2507]

resid = spec - model
chisq = np.sum( (resid/errs)**2)
print("chisq is ",chisq," for ",len(resid)-len(pars)," degrees of freedom.")