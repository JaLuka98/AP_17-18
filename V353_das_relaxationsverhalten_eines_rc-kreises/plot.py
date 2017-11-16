import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def linfit(x, a, b):
    return x*a+b


def amplitudefit(x, RC):
    return 1/(np.sqrt(1+(4*x*(np.pi))**2*(RC)**2))


def arctanfit(x, d):
    return np.arctan(2*np.pi * d*x)/np.pi/2 * 360

# Entladung des Kondensators

Uc, t = np.genfromtxt('data/abfallende_flanke.txt', unpack=True)  # Uc in mV, t in ms
U0 = 1090
Uc[0] = 1088
params, covariance_matrix = optimize.curve_fit(linfit, t, np.log(Uc/U0))

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

print('RC =', (-1/a)*1e-3)

plt.plot(t, np.log(Uc/U0), 'rx', label='Messwerte')
plt.plot(t, linfit(t, *params), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$t/$ms')
plt.ylabel(r'$\ln{\left(\frac{U_\mathrm{C}}{U_0}\right)}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/uc.pdf')

plt.clf()

# Berücksichtigung der Frequenzabhängigkeit der Amplitude von Uc

f, Uc = np.genfromtxt('data/amplitude.txt', unpack=True)  # Uc in mV, f in Hz
U0 = 503.68  # in mV
U_normiert = Uc/U0

params2, covariance_matrix2 = optimize.curve_fit(amplitudefit, f, U_normiert, p0 = 0.000001)

errors2 = np.sqrt(np.diag(covariance_matrix2))

print('RC =', params2[0], '+-', errors2[0])

RC = ufloat(params2[0], errors2[0])

frequenz = np.linspace(20, 5000, 5000)

plt.plot(f, U_normiert, 'rx', label='Messwerte')
plt.plot(frequenz, amplitudefit(frequenz, params2[0]), 'k-', label='Ausgleichsfunktion')
plt.xscale('log')
plt.xlabel(r'$f/$Hz')
plt.ylabel(r'$U_normiert/$mV')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/amplitude.pdf')

plt.clf()

# Berücksichtigung der Frequenzabhängigkeit der Phasenverschiebung von Uc zu U0

f, a = np.genfromtxt('data/phasenverschiebung.txt', unpack=True)  # f in Hz, a in ms

a = a * 1e-3  # a in s

phi = a*f*360

params3, covariance_matrix3 = optimize.curve_fit(arctanfit, f, phi)

errors3 = np.sqrt(np.diag(covariance_matrix3))

print('RC =', params3[0], '+-', errors3[0])

RC = ufloat(params3[0], errors3[0])

frequenz = np.linspace(20, 5000, 5000)

plt.plot(f, phi, 'rx', label='Messwerte')
plt.plot(f, arctanfit(f, params3[0]), 'k-', label='Ausgleichsfunktion')
plt.xscale('log')
plt.xlabel(r'$f/$Hz')
plt.ylabel(r'$phi/$grad')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/phi.pdf')

plt.clf()

# Und der Polarplot

plt.figure(3)
phirange = np.linspace(0, np.pi/2, 10000)
plt.polar(phi/360*2*np.pi, U_normiert, "r+", label="Messung")
plt.polar(phirange, np.cos(phirange), "b-", label="Theorie")
plt.legend(loc="best")
plt.savefig("build/polar.pdf")
