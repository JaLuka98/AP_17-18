import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex


#Bragg

alpha, I = np.genfromtxt('data/bragg.txt', unpack=True)
plt.plot(alpha, I, 'rx', mew=0.5)
plt.xlabel(r'$\alpha/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/bragg.pdf')
plt.clf()

# hr = ['$\alpha/$Grad', '$I$']
# m = np.zeros((np.size(I), 2))
# m[: ,0] = alpha
# m[:, 1] = I
# t = matrix2latex(m, headerRow=hr, format='%.1f')
# print(t)

#Emission

zweitheta, I = np.genfromtxt('data/bragg.txt', unpack=True)
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/emission.pdf')
plt.clf()

hr = ['$2 \cdot \theta/$Grad', '$I$']
m = np.zeros((np.size(I), 2))
m[:, 0] = zweitheta
m[:, 1] = I
t = matrix2latex(m, headerRow=hr, format='%.1f')
print(t)

#Strontium

zweitheta, I = np.genfromtxt('data/strontium.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/strontium.pdf')
plt.clf()

#Zirkonium

zweitheta, I = np.genfromtxt('data/zirkonium.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zirkonium.pdf')
plt.clf()

#Zink

zweitheta, I = np.genfromtxt('data/zink.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zink.pdf')
plt.clf()

#Brom

zweitheta, I = np.genfromtxt('data/brom.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/brom.pdf')
plt.clf()

#Gold

zweitheta, I = np.genfromtxt('data/gold.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/gold.pdf')
plt.clf()
