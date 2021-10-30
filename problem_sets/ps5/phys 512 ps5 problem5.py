"""
Jules Faucher
260926201

Phys 512
October 29th, 2021
"""

import numpy as np
import matplotlib.pyplot as plt

def dft(k, N): #DFT as derived in the pdf
    
    dft = np.zeros(N, dtype = 'complex')

    for i in range(0,N):
        dft[i] = ((1 - np.exp(-2j*np.pi*(i - k)))/(1 - np.exp(-2j*np.pi*(i - k)/N)) - (1 - np.exp(-2j*np.pi*(i + k)))/(1 - np.exp(-2j*np.pi*(i + k)/N)))/2j
    
    return dft

N = 100

x = np.arange(N)
k = 20.33
y = np.sin(2*np.pi*k*x/N)


dft = dft(k, N)

fft = np.fft.fft(y)

plt.plot(dft, label = 'DFT')
plt.plot(fft, '--', label = 'FFT')
plt.title('DFT and FFT of sine', fontsize = 16)
plt.xlabel('k', fontsize = 14)
plt.ylabel('F(k)', fontsize = 14)
plt.savefig('./problem5c.png', dpi = 300)
plt.legend()
plt.grid()
plt.show()

plt.plot(dft- fft)
plt.title('Residuals', fontsize = 16)
plt.xlabel('k', fontsize = 14)
plt.ylabel('F(k)', fontsize = 14)
plt.savefig('./problem5c_residuals.png', dpi = 300)
plt.grid()
plt.show()


#Part d
window = 0.5 - 0.5*np.cos(2*np.pi*x/N)
fft_wind = np.fft.fft(y*window)

plt.plot(fft, label = 'FFT')
plt.plot(fft_wind, label = 'Windowed FFT')
plt.title('Windowed FFT and Regular FFT', fontsize = 16)
plt.ylabel('F(k)', fontsize = 14)
plt.xlabel('k', fontsize = 14)
plt.savefig('./Problem5d.png', dpi = 300)
plt.legend()
plt.grid()
plt.show()

#Part e
fft_window = np.fft.fft(window)

expected = np.zeros(len(fft_window))
expected[0] = N/2
expected[1] = -N/4
expected[-1] = -N/4


plt.plot((fft_window), label = 'FFT of Window')
plt.plot(expected, '--', label = 'Expectation')
plt.title('FFT of the Window', fontsize = 16)
plt.ylabel('F(k)', fontsize = 14)
plt.xlabel('k', fontsize = 14)
plt.savefig('./Problem5e.png')
plt.legend()
plt.grid()
plt.show()

plt.plot((fft_window)- expected)
plt.title('Residuals', fontsize = 16)
plt.ylabel('F(k)', fontsize = 14)
plt.xlabel('k', fontsize = 14)
plt.savefig('./Problem5e_resid.png')
plt.grid()
plt.show()

fft_y = np.fft.fft(y)
fft_combinations = 0.5*fft_y - 0.25*np.roll(fft_y,1) - 0.25*np.roll(fft_y, -1)

plt.plot(fft_combinations, label = 'FFT Combo')
plt.plot(fft_wind, label = 'Windowed FFT', ls = '--')

plt.title('Combinations and Windowed FFT', fontsize = 16)
plt.ylabel('F(k)', fontsize = 14)
plt.xlabel('k', fontsize = 14)
plt.savefig('./fft_combo_and_window.png', dpi = 300)
plt.legend()
plt.grid()
plt.show()
