import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def einhuellendenfit(t, A0, my):
    return A0*np.exp(-2*np.pi*my*t)


# L=(10.11±0.03)*10^-3H
# C=(2.098±0.006)*10^-9F
# R_1=(48.1±0.1)Ω
# R_2=(509.5±0.5)Ω

# 4b)R=3350Ω

L = ufloat(10.11, 0.03)  # in mH
L *= 1e-3  # in H
C = ufloat(2.098, 0.006)  # in nF
C *= 1e-9  # in F
R1 = ufloat(48.1, 0.1)  # in Ohm
R2 = ufloat(509.5, 0.5)  # in Ohm

t, Umal10 = np.genfromtxt('data/einhuellende.txt', unpack=True)  # t in µs, Umal10 in V
U = Umal10/10
t *= 1e-6  # t in s

params, covariance_matrix = optimize.curve_fit(einhuellendenfit, t, U)

errors = np.sqrt(np.diag(covariance_matrix))

A0 = ufloat(params[0], errors[0])
my = ufloat(params[1], errors[1])

print('A0 =', params[0], '+-', errors[0])
print('my =', params[1], '+-', errors[1])

print('Dann ist Reff aus dem Experiment ja ', 4*my*np.pi*L)
print('Tex ist hier dann ', (1/(2*np.pi*my))*1e6, 'in µs')

plt.plot(t*1e6, U, 'rx', label='Messwerte')
plt.plot(t*1e6, einhuellendenfit(t*1e6, params[0], params[1]*1e-6), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$t/$µs')
plt.ylabel(r'$U_\mathrm{C}/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/einhuellende.pdf')

plt.clf()

print('Rap nach der Theorie ist ', unp.sqrt(4*L/C))
print('Der absolute Fehler ist', 3550 - unp.sqrt(4*L/C))

f, Ucmal10, U = np.genfromtxt('data/amplitude.txt', unpack=True)  # f in Hz, Ucmal10, U in V
Uc = Ucmal10/10
U = U/10
f *= 1e-3  # f in khZ

np.savetxt('data/amplitudeucu.txt', Uc/U, delimiter='\t')

plt.plot(f, Uc/U, 'rx', label='Messwerte')
plt.xlabel(r'$f/$kHz')
plt.ylabel(r'$\frac{U_\mathrm{C}}{U_0}$')
plt.xscale('log')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/amplitudelog.pdf')

print('Die Güte q ergibt sich theoretisch zu', 1/(R2+50)*unp.sqrt(L/C))  # 50: Berücksichtigung des Innenwiderstandes

plt.xscale('linear')
plt.axis((10000*1e-3, 46000*1e-3, 1, 4))  # 1e-3 weil f in kHZ ist
plt.axhline(2.6502, color='k', linestyle=':')
plt.axvline(28250*1e-3, color='b', linestyle=':', label=r'$f_{+}$ und $f_{-}$')
plt.axvline(37500*1e-3, color='b', linestyle=':')
plt.legend()
plt.savefig('build/amplitudelin.pdf')

print('Die Breite der Resonanzkurve ergibt sich theoretisch zu', (R2+50)/(2*np.pi*L))

plt.clf()

f, a = np.genfromtxt('data/phasenverschiebung.txt', unpack=True)  # f in Hz, a in µs
a *= 1e-6  # a in s
phi = a*f*360
phi = phi/360*2*np.pi
f *= 1e-3  # f in khZ

plt.plot(f, phi, 'rx', label='Messwerte')
plt.xscale('log')
plt.xlabel(r'$f/$kHz')
plt.ylabel(r'$\phi/$rad')
# Slicing: -3 ist drittletztes Element, : bedeutet bis zum Ende
plt.plot(f[-3:], phi[-3:], 'ob', markersize=8, markeredgewidth=1, markerfacecolor='None' )  # Sorgt für Markierung dr Außreißer

# Macht die y-Achse schön
x = np.linspace(0, 2 * np.pi)

plt.ylim(0, 1.25 * np.pi)
# erste Liste: Tick-Positionen, zweite Liste: Tick-Beschriftung
plt.yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi/4, np.pi],
           [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$"])

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/philog.pdf')

plt.xscale('linear')
plt.axis((20000*1e-3, 80000*1e-3, 0, np.pi))
plt.axvline(30000*1e-3, color='k', linestyle=':', label=r'$f_1$ und $f_2$')
plt.axvline(38150*1e-3, color='k', linestyle=':')
plt.axvline(33800*1e-3, color='b', linestyle=':', label='Resonanzfrequenz')
plt.legend()
plt.savefig('build/philin.pdf')

plt.clf()
