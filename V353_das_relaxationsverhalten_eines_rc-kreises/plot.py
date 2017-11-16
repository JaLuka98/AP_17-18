import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def f(x, a, b):
    return x*a+b


# Entladung des Kondensators

Uc, t = np.genfromtxt('data/abfallende_flanke.txt', unpack=True)  # Uc in mV, t in ms
U0 = 1090
Uc[0] = 1088
params, covariance_matrix = optimize.curve_fit(f, t, np.log(Uc/U0))

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(t, np.log(Uc/U0), 'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$t/$ms')
plt.ylabel(r'$\ln{\left(\frac{U_\mathrm{C}}{U_0}\right)}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/uc.pdf')

plt.clf()

# Berücksichtigung der Frequenzabhängigkeit der Amplitude von Uc

f, Uc = np.genfromtxt('data/amplitude.txt', unpack=True) # Uc in mV, f in Hz
U0 = 1


plt.plot(f, Uc, 'rx', label='Messwerte')
plt.xscale('log')
plt.xlabel(r'$f/$Hz')
plt.ylabel(r'$U/$mV')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/amplitude.pdf')
