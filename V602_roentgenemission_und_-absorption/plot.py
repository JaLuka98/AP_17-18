import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex

def linfit(x,a,b):
    return a*x+b

#Bragg

alpha, I = np.genfromtxt('data/bragg.txt', unpack=True)
plt.plot(alpha, I, 'rx', mew=0.5)
plt.xlabel(r'$\alpha/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/bragg.pdf')
plt.clf()

# hr = ['$\alpha/$°', '$I$']
# m = np.zeros((np.size(I), 2))
# m[: ,0] = alpha
# m[:, 1] = I
# t = matrix2latex(m, headerRow=hr, format='%.1f')
# print(t)

#Emission

zweitheta, I = np.genfromtxt('data/emission.txt', unpack=True)
zweitheta[0] = 8.0 # dummer scheiß geht nur so
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/emission.pdf')
plt.clf()

plt.subplot(1, 2, 1)
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.plot(zweitheta, I, 'b-', linewidth=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.xlim(38, 42)
plt.ylim(0, 1100)
plt.axhline(1021, color='b', linestyle='--', linewidth=0.5)
plt.axhline(85, color='b', linestyle='--', linewidth=0.5)
plt.axhline(468, color='g', linestyle='--', linewidth=0.5)
plt.axvline(39.69, color='m', linestyle='--', linewidth=0.5)
plt.axvline(40.555, color='m', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.plot(zweitheta, I, 'b-', linewidth=0.5)
plt.xlabel(r'$2 \cdot \theta/$Grad')
plt.ylabel(r'$I$')
plt.xlim(42.5, 46.5)
plt.ylim(0, 3500)
plt.axhline(3195, color='b', linestyle='--', linewidth=0.5)
plt.axhline(86, color='b', linestyle='--', linewidth=0.5)
plt.axhline(1554.5, color='g', linestyle='--', linewidth=0.5)
plt.axvline(44.18, color='m', linestyle='--', linewidth=0.5)
plt.axvline(45.04, color='m', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.grid()

plt.savefig('build/peaks.pdf')
plt.clf()


hr = ['$2 \cdot \theta/$°', '$I$']
m = np.zeros((np.size(alpha), 4))
m[:, 0] = zweitheta[0:np.size(alpha)]
m[:, 1] = I[0:np.size(alpha)]
m[:, 2] = zweitheta[np.size(alpha):2*np.size(alpha)]
m[:, 3] = I[np.size(alpha):2*np.size(alpha)]
#temp = np.zeros(np.size(alpha))
#for i in range(0, np.size(zweitheta)-2*np.size(alpha)):
#    temp += zweitheta[2*np.size(alpha)+i]
#print(temp)
#m[:, 4] = np.zeros(np.size(alpha)) + zweitheta
#m[:, 5] = I[2*np.size(alpha):-1]
t = matrix2latex(m, headerRow=hr, format='%.1f')
print(t)
#Shits not even close to working

#Strontium

zweitheta, I = np.genfromtxt('data/strontium.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/strontium.pdf')
plt.clf()

#Zirkonium

zweitheta, I = np.genfromtxt('data/zirkonium.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zirkonium.pdf')
plt.clf()

#Zink

zweitheta, I = np.genfromtxt('data/zink.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis((31.75, 39.3, 27.5, 60))
plt.savefig('build/zink.pdf')
plt.clf()

#Brom

zweitheta, I = np.genfromtxt('data/brom.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/brom.pdf')
plt.clf()

# Bestimmung der Rydbergenergie durch Moseley

Z = [30,35,38,40]
zlinspace = np.linspace(20, 50, 500)
E_k = [9.575856772, 13.42955074, 15.84713652, 17.05114588]
params, covariance_matrix = optimize.curve_fit(linfit, Z, np.sqrt(E_k))

a, b = correlated_values(params, covariance_matrix)

print('Fit zur Rydbergenergie')
print('a =', a)
print('a^2 =', a**2*1e3)
print('b =', b)

plt.plot(Z, np.sqrt(E_k), 'rx', mew=0.5)
plt.plot(zlinspace, linfit(zlinspace, *params), 'r-')
plt.xlabel(r'$Z$')
plt.ylabel(r'$\sqrt{E_K}/\mathrm{\sqrt{keV}}')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis((27.5, 42.5, 2.5, 4.75))
plt.savefig('build/rydberg.pdf')
plt.clf()

#Gold

zweitheta, I = np.genfromtxt('data/gold.txt', unpack=True) #f/kHz, U/V
plt.plot(zweitheta, I, 'rx', mew=0.5)
plt.xlabel(r'$2 \cdot \theta/$°')
plt.ylabel(r'$I$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/gold.pdf')
plt.clf()
