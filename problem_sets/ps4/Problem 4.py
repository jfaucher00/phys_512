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


def get_spectrum(pars,lmax=3000):
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
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    return tt[2:][0:2507]


def Grad(fun, x):
    dim = len(x)
    h = 1e-8
    hI = h*np.eye(dim)*pars
    copies = np.outer(x, np.ones(dim)).T
    xplush = copies + hI
    xminush = copies - hI
    grad = []
    for i in range(dim):
        grad += [(fun(xplush[i]) - fun(xminush[i]))/2/h/pars[i]]
    return np.array(grad)


def fit_newton(m, fun, y, errs, max_iter = 10, chi_tol = 0.02):
    
    len_y = len(y)
    current_iter = 0
    chi1 = 4*len_y
    chi2 = 3*len_y
    
    Ninv = np.eye(len_y)/(errs**2)
    
    while current_iter < max_iter and (chi2 + chi_tol < chi1):
        
        f = fun(m)[0:len_y]
        r = y - f
        pre_df = Grad(fun, m)
        df = pre_df.T[0:len_y].T
        
        lhs = df@Ninv@df.T
        rhs = df@Ninv@r.T
        cov = np.linalg.inv(lhs)
        dm = cov@rhs
        m += dm
        
        chi1 = chi2
        chi2 = np.sum( (r/errs)**2)
        current_iter += 1
        
    pars_errs = np.sqrt(np.diag(cov))    
    return m, pars_errs, cov

def spec_chi(pars, y, noise):
    model = get_spectrum(pars)
    return np.sum(((y - model)/noise)**2)

def prior_chisq(pars, par_errs): #Function that returns an estimate of the past chi2
    par_shifts = pars - par_errs
    return np.sum((par_shifts/par_errs)**2)


def mcmc(pars, step_size, y, fun, nstep = 2000, noise = 1, par_priors = None, par_errs = None):
    
    chi_cur = fun(pars, y, noise) + prior_chisq(pars, par_priors, par_errs)
    npar = len(pars)
    chain = np.zeros([nstep,npar])
    chivec = np.zeros(nstep)
    
    for i in range(nstep):
        trial_pars = pars + step_size @ np.random.randn(npar)/10
        trial_chisq = fun(trial_pars, y, noise) + prior_chisq(pars, par_priors, par_errs)
        delta_chisq = trial_chisq - chi_cur
        accept_prob = np.exp(-0.5*delta_chisq)
        
        if np.random.rand(1) < accept_prob:
            pars = trial_pars
            chi_cur = trial_chisq
        chain[i,:] = pars
        chivec[i] = chi_cur
        print(i//(nstep//100), "%")
        #print(time.time() - t1)
    return chain, chivec


pars=np.asarray([69, 0.022, 0.12, 0.06, 2.1e-9, 0.95])

planck = np.loadtxt(dirname + '/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
spec = planck[:,1]
errs = 0.5*(planck[:,2] + planck[:,3])

chain_and_chisq = np.loadtxt(dirname + "/planck_chain.txt") #Reusing the chains from p3
chain = chain_and_chisq.T[1:].T
chisq = chain_and_chisq[:,0]

chain = chain[1000:] #Since they had not converged, I use the last half only
chisq = chisq[1000:]

step_size_new = np.std(chain, axis = 0)
starting_pars = np.mean(chain, axis = 0) + 3*np.random.randn(len(pars))*step_size_new
starting_pars[3] = 0.054 + 3*np.random.randn(1)*0.0074

fit_pars2, fit_errs, cov2 = fit_newton(starting_pars, get_spectrum, spec, errs)
cholesky2 = np.linalg.cholesky(cov2) #Getting new curvature matrix

#Running mcmc a second time
chain2, chisq2 = mcmc(starting_pars, cholesky2, spec, spec_chi, noise = errs)


chain2 = chain2[1000:] #Once again the chains have not converged so I used
chisq2 = chisq2[1000:] #the later half to give myself a chance

nsamp = chain2.shape[0]
weight = np.zeros(nsamp)
chivec = np.zeros(nsamp)

for i in range(nsamp):
    chisq = prior_chisq(chain[i], fit_pars2, step_size_new)
    chivec[i] = chisq
chivec = chivec - chivec.mean()
weight = np.exp(0.5*chivec)

#The weighting doesn't work properly without convergence

final_pars = np.sum(np.diag(weight)@chain2, axis = 0)/np.sum(weight)


plt.loglog(np.abs(np.fft.rfft(chain2[:,0])), label = "$H_0$")
plt.loglog(np.abs(np.fft.rfft(chain2[:,1])), label = "$\Omega_b h^2$")
plt.loglog(np.abs(np.fft.rfft(chain2[:,2])), label = "$\Omega_C h^2$")
plt.loglog(np.abs(np.fft.rfft(chain2[:,3])), label = "$\tau$")
plt.loglog(np.abs(np.fft.rfft(chain2[:,4])), label = "$A_s$")
plt.loglog(np.abs(np.fft.rfft(chain2[:,5])), label = "$n_s$")
plt.grid()
plt.title("Fourier Series of the Sampled Chains", fontsize = 16)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Parameter Value", fontsize = 14)
plt.legend()
plt.show()

plt.loglog(chisq2, label = "$\chi^2$")
plt.grid()
plt.title("$\chi^2$ as Function of Iteration", fontsize = 16)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("$\chi^2$", fontsize = 14)
plt.legend()
plt.show()

fit_pars = np.mean(chain2, axis = 0)
fit_errs = np.std(chain2, axis = 0)

Omega_b = fit_pars[1]*100**2/(fit_pars[0]**2)
Omega_c = fit_pars[2]*100**2/(fit_pars[0]**2)
Omega_a = 1 - Omega_b - Omega_c