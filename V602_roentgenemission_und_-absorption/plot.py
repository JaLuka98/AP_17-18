import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex


#Bragg

I, alpha = np.genfromtxt('data/bragg.txt', unpack=True) #f/kHz, U/V
plt.plot(I, alpha, 'rx', mew=0.5)
plt.xlabel(r'$\alpha/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/bragg.pdf')
plt.clf()

#Emission

I, zweitheta = np.genfromtxt('data/emission.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/emission.pdf')
plt.clf()

#Strontium

I, zweitheta = np.genfromtxt('data/strontium.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/strontium.pdf')
plt.clf()

#Zirkonium

I, zweitheta = np.genfromtxt('data/zirkonium.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zirkonium.pdf')
plt.clf()

#Zink

I, zweitheta = np.genfromtxt('data/zink.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zink.pdf')
plt.clf()

#Brom

I, zweitheta = np.genfromtxt('data/brom.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/brom.pdf')
plt.clf()

#Gold

I, zweitheta = np.genfromtxt('data/gold.txt', unpack=True) #f/kHz, U/V
plt.plot(I, zweitheta, 'rx', mew=0.5)
plt.xlabel(r'$2\theta/$°')
plt.ylabel(r'$I/$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/gold.pdf')
plt.clf()
