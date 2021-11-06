"""
Jules Faucher
260926201

Phys 512
November 5, 2021
"""

import h5py
import json
import readligo as rl
import numpy as np
from scipy import signal
import scipy.interpolate as sci
import scipy.signal as scs
import matplotlib.pyplot as plt
import os

files = os.listdir()
templates = [e for e in files if e[-13:] == "template.hdf5"]
H_files = [e for e in files if e[:9] == "H-H1_LOSC"]
L_files = [e for e in files if e[:9] == "L-L1_LOSC"]


def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    th=template[0]
    tl=template[1]
    return th,tl

def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

def flat_cos(n, m):
    x=np.linspace(-np.pi,np.pi,m)
    tips = 0.5 + 0.5*np.cos(x)
    flat = np.ones(n)
    half_m = m//2
    flat[:half_m] = tips[:half_m]
    flat[-half_m:] = tips[-half_m:]
    return flat

def planck(N, eps = 0.1): #Planck-taper window

    flat = np.ones(N)
    flat[0] = 0
    flat[-1] = 0
    n = np.arange(1, N*eps-1)
    slope1 = 1/(1 + np.exp((eps*N)/n - eps*N/(eps*N-n)))
    slope2 = slope1[::-1]
    flat[1: int(N*eps)] = slope1
    flat[-int(N*eps): -1] = slope2
    
    return flat

Gaussian = lambda x, mu, sigma : np.exp( -0.5*((x-mu)/sigma)**2 )/np.sqrt(2*np.pi)/sigma

strainL, dtL, utcL = read_file(L_files[0])
strainH, dtH, utcH = read_file(H_files[0])
th, tl = read_template(templates[0])



th_len = len(th)

wind = planck(th_len, eps = 0.1) #window
temp_spec = np.abs(np.fft.fft(th))**2
temp_wind = th*wind
#temp_wind_spec = np.abs(np.fft.fft(temp_wind))**2
#plt.loglog(temp_spec)
#plt.loglog(temp_wind_spec)


n = len(strainH)
ns = np.linspace(0, 2048, n)
sft = np.fft.fft(wind*strainH)

Nft = np.abs(sft)**2 #Power spectrum
Area1_Nft = np.sum(Nft)
plt.loglog(Nft)

for i in range(10):
    Nft = np.fft.ifft( np.fft.fft(Gaussian(ns, 0, 0.1)) * np.fft.fft(Nft) )

Area2_Nft = np.sum(Nft)
Nft = Nft*Area1_Nft/Area2_Nft

plt.loglog(Nft)

#sft_white = sft/np.sqrt(Nft_white)
#plt.loglog(sft_white)
#tft_white = np.fft.rfft(th*wind)/np.sqrt(Nft)
#t_white = np.fft.irfft(tft_white)

#plt.loglog(Nft)