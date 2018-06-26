import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from scipy import stats


def correct(x, A0, A2):
    return A0 + A2/x**2


def incorrect(x, A0strich, A2strich):
    return A0strich - A2strich*x**2


lammda = np.array([643.84695, 614.9475, 614.6435, 579.067, 546.075, 508.58217, 467.81493, 435.8335, 404.6565])
omega_r, omega_l = np.genfromtxt("data/eta.txt", unpack='true')
phi_l, phi_r = np.genfromtxt("data/phi.txt", unpack='true')

eta = np.abs(180 - (omega_r - omega_l))
eta[0] = 60.1
phi = (phi_r-phi_l)/2
print(eta)
print(phi)
eta = np.deg2rad(eta)
phi = np.deg2rad(phi)
phi = ufloat(np.mean(phi), stats.sem(phi))


n = unp.sin((eta+phi)/2)/unp.sin(phi/2)

# korrekte Funktion
params, covariance_matrix = optimize.curve_fit(correct, lammda, unp.nominal_values(n**2))
A0, A2 = correlated_values(params, covariance_matrix)
print('A0 = ', A0)
print('A2 = ', A2)
lammdalinspace = np.linspace(350, 700, 500)

# Nicht so korrekte Funktion
params2, covariance_matrix2 = optimize.curve_fit(incorrect, lammda, unp.nominal_values(n**2))
A0strich, A2strich = correlated_values(params2, covariance_matrix2)
print('A0strich = ', A0)
print('A2strich = ', A2)
lammdalinspace = np.linspace(350, 700, 500)

plt.plot(lammda,unp.nominal_values(n**2), 'rx', label='Messwerte', mew=0.5)
plt.plot(lammdalinspace, correct(lammdalinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)
#plt.plot(lammdalinspace, incorrect(lammdalinspace, *params2), 'r-', label='Ausgleichsfunktion', linewidth=0.5)
plt.xlabel(r'$\lambda/$nm')
plt.ylabel(r'$n^2$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([350, 700, 2.95, 3.25])
plt.savefig('build/kurve.pdf')
plt.clf()

skorrektquadrat = 0
for i in range(0, np.size(unp.nominal_values(n**2))):
    skorrektquadrat += (unp.nominal_values(n[i]**2) - A0 - A2/lammda[i]**2)**2
skorrektquadrat *= 1/(np.size(unp.nominal_values(n**2))-2)
print(unp.nominal_values(skorrektquadrat))
print(unp.std_devs(skorrektquadrat))

sinkorrektquadrat = 0
for i in range(0, np.size(unp.nominal_values(n**2))):
    sinkorrektquadrat += (unp.nominal_values(n[i]**2) - A0strich - A2strich/lammda[i]**2)**2
sinkorrektquadrat *= 1/(np.size(unp.nominal_values(n**2))-2)
print(unp.nominal_values(sinkorrektquadrat))
print(unp.std_devs(sinkorrektquadrat))
