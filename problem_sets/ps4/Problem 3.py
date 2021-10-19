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

dirname = os.path.dirname(__file__) + "\\"

#Curvature matrix taken from Problem 2
curva = np.array([[ 1.68806012e-01,  1.56292295e-05, -3.75257160e-04,
                   8.45179645e-03,  3.48671074e-11,  2.27253416e-03],
                  [ 1.56292295e-05,  3.61530500e-08,  4.91041439e-08,
                   2.98281048e-06,  1.35324211e-14,  5.04061197e-07],
                  [-3.75257160e-04,  4.91041439e-08,  1.21200358e-06,
                   -1.55459742e-05, -6.04840934e-14, -4.87811732e-06],
                  [ 8.45179645e-03,  2.98281048e-06, -1.55459742e-05,
                   1.51675924e-03,  6.37397563e-12,  1.46455805e-04],
                  [ 3.48671074e-11,  1.35324211e-14, -6.04840934e-14,
                   6.37397563e-12,  2.68412543e-20,  6.05128180e-13],
                  [ 2.27253416e-03,  5.04061197e-07, -4.87811732e-06,
                   1.46455805e-04,  6.05128180e-13,  4.23841606e-05]])

cholesky = np.linalg.cholesky(curva)

def get_spectrum(pars,lmax=3000):
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau = pars[3]
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
    return tt[2:][0:2507]

def spec_chi(pars, y, noise):
    model = get_spectrum(pars)
    return np.sum(((y - model)/noise)**2)

def mcmc(pars, y, fun, nstep = 2000, noise = 1):
    
    chi_cur = fun(pars, y, noise)
    npar = len(pars)
    chain=np.zeros([nstep,npar])
    chivec=np.zeros(nstep)
    
    for i in range(nstep):
        t1 = time.time()
        trial_pars = pars + cholesky @ np.random.randn(npar) #Steps are scaled 
        trial_chisq = fun(trial_pars, y, noise)     #with the curvature matrix
        delta_chisq = trial_chisq - chi_cur
        accept_prob = np.exp(-0.5*delta_chisq)
        
        if np.random.rand(1) < accept_prob:
            pars = trial_pars
            chi_cur = trial_chisq
        chain[i,:] = pars
        chivec[i] = chi_cur
        print(i//(nstep//100), "%")
        print(time.time() - t1)
    return chain, chivec


pars = np.asarray([69, 0.022, 0.12, 0.08, 2.1e-9, 0.95])

planck = np.loadtxt(dirname + '/COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
spec = planck[:,1]
errs = 0.5*(planck[:,2]+planck[:,3])

chain , chisq = mcmc(pars, spec, spec_chi, noise = errs)

plt.loglog(np.abs(np.fft.rfft(chain[:,0])), label = "$H_0$")
plt.loglog(np.abs(np.fft.rfft(chain[:,1])), label = "$\Omega_b h^2$")
plt.loglog(np.abs(np.fft.rfft(chain[:,2])), label = "$\Omega_C h^2$")
plt.loglog(np.abs(np.fft.rfft(chain[:,3])), label = "$A_s$")
plt.loglog(np.abs(np.fft.rfft(chain[:,4])), label = "$n_s$")
plt.grid()
plt.title("Fourier Series of the Chains", fontsize = 16)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Parameter Value", fontsize = 14)
plt.legend()
plt.show()

plt.loglog(chisq, label = "$\chi^2$")
plt.grid()
plt.title("$\chi^2$ as Function of Iteration", fontsize = 16)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("$\chi^2$", fontsize = 14)
plt.legend()
plt.show()

fit_pars = np.mean(chain, axis = 0)
fit_errs = np.std(chain, axis = 0)

Omega_b = fit_pars[1]*100**2/(fit_pars[0]**2)
Omega_c = fit_pars[2]*100**2/(fit_pars[0]**2)
Omega_a = 1 - Omega_b - Omega_c

O_b_err = Omega_b*np.sqrt((fit_errs[1]/fit_pars[1])**2+(2*fit_errs[0]/fit_pars[0])**2)
O_c_err = Omega_c*np.sqrt((fit_errs[2]/fit_pars[2])**2+(2*fit_errs[0]/fit_pars[0])**2)
O_a_err = np.sqrt(O_b_err**2 + O_c_err**2)

to_save = np.hstack((np.reshape(chisq, (2000, 1)), chain))
header = """The chain did not converge since the Fourier transform plot is not flat.
{Omega}({err})% of the universe is made out of dark energy.
Complete values = {OmegaC} +/- {errC}
""".format(Omega = round(Omega_a*100), err = round(O_a_err*100), OmegaC = Omega_a, errC = O_a_err)
np.savetxt(dirname + "planck_chain.txt", to_save, header = header)