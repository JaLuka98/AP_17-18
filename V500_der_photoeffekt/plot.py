import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex


def linearfit(x, a, b):
    return a*x+b


gelbU, gelbI = np.genfromtxt('data/gruen.txt', unpack=True)
gruenU, gruenI = np.genfromtxt('data/gelb.txt', unpack=True)
gruenblauU, gruenblauI = np.genfromtxt('data/gruenblau.txt', unpack=True)
violett1U, violett1I = np.genfromtxt('data/violett1.txt', unpack=True)
violett2U, violett2I = np.genfromtxt('data/violett2.txt', unpack=True)
ultraviolettU, ultraviolettI = np.genfromtxt('data/ultraviolett.txt', unpack=True)

# Die gelbe (orangefarbene) Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gelbU, np.sqrt(gelbI))
a, b = correlated_values(params, covariance_matrix)
print('U_g für gelb =', -b/a)
gelblinspace = np.linspace(-0.1, 0.7, 500)

plt.plot(gelbU, np.sqrt(gelbI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gelblinspace, linearfit(gelblinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.7, -1, 18])
plt.savefig('build/gelb.pdf')
plt.clf()

# Die gruene Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gruenU, np.sqrt(gruenI))
a, b = correlated_values(params, covariance_matrix)
print('U_g für grün =', -b/a)
gruenlinspace = np.linspace(-0.1, 0.7, 500)

plt.plot(gruenU, np.sqrt(gruenI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gruenlinspace, linearfit(gruenlinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.6, -1, 10])
plt.savefig('build/gruen.pdf')
plt.clf()

# Die gruenblaue Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gruenblauU, np.sqrt(gruenblauI))
a, b = correlated_values(params, covariance_matrix)
print('U_g für gruenblau =', -b/a)
gruenblaulinspace = np.linspace(-0.1, 0.9, 500)

plt.plot(gruenblauU, np.sqrt(gruenblauI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gruenblaulinspace, linearfit(gruenblaulinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.90, -1, 6])
plt.savefig('build/gruenblau.pdf')
plt.clf()

# Die erste violette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, violett1U, np.sqrt(violett1I))
a, b = correlated_values(params, covariance_matrix)
print('U_g für erste violette =', -b/a)
violett1linspace = np.linspace(-0.1, 1.35, 500)

plt.plot(violett1U, np.sqrt(violett1I), 'rx', label='Messwerte', mew=0.5)
plt.plot(violett1linspace, linearfit(violett1linspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.3, -1, 26])
plt.savefig('build/violett1.pdf')
plt.clf()

# Die zweite violette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, violett2U, np.sqrt(violett2I))
a, b = correlated_values(params, covariance_matrix)
print('U_g für zweite violette =', -b/a)
violett2linspace = np.linspace(-0.1, 1.5, 500)

plt.plot(violett2U, np.sqrt(violett2I), 'rx', label='Messwerte', mew=0.5)
plt.plot(violett2linspace, linearfit(violett2linspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.5, -1, 16.5])
plt.savefig('build/violett2.pdf')
plt.clf()

# Die ultraviolette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, ultraviolettU, np.sqrt(ultraviolettI))
a, b = correlated_values(params, covariance_matrix)
print('U_g für ultraviolette =', -b/a)
ultraviolettlinspace = np.linspace(-0.1, 1.95, 500)

plt.plot(ultraviolettU, np.sqrt(ultraviolettI), 'rx', label='Messwerte', mew=0.5)
plt.plot(ultraviolettlinspace, linearfit(ultraviolettlinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.8, -1, 19])
plt.savefig('build/ultraviolett.pdf')
plt.clf()
