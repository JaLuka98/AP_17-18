import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from scipy.constants import codata


#rührmotoren: 5V
#1  wärmekap. reservoire<. 750 J/K
#V=4l

def f(x, a, b, c):
    return a*x**2+b*x+c

def g(x, a, b):
    return -x*a/codata.value('molar gas constant')+b



t, T1, T2, p_a, p_b, P = np.genfromtxt('data/data.txt', unpack=True) #t/min T1/°C T2/°C p_a/bar p_b/bar P/Watt

p_a+=1  #Umgebungsdruck
p_b+=1
t*=60   #t in s
T1+=273.15
T2+=273.15


np.savetxt('umgerechnet.txt', np.column_stack([t, T1, T2, p_a, p_b, P, 1/T1*1e3, np.log(p_b)]))


params, covariance_matrix = optimize.curve_fit(f, t, T1)
errors = np.sqrt(np.diag(covariance_matrix))
print('a1 = ', params[0], '+-', errors[0])
print('b1 = ', params[1], '+-', errors[1])
print('c1 = ', params[2], '+-', errors[2])
a1 = ufloat(params[0], errors[0])
b1 = ufloat(params[1], errors[1])
c1 = ufloat(params[2], errors[2])
print('Differential300 = ', 2*300*a1+b1)
print('Differential900 = ', 2*900*a1+b1)
print('Differential1200 = ', 2*1200*a1+b1)
print('Differential1800 = ', 2*1800*a1+b1)
tlinspace = np.linspace(0, max(t)+300, 5000)
print('nu300 = ', (2*300*a1+b1)*(16748+750)*1/124)
print('nu900 = ', (2*900*a1+b1)*(16748+750)*1/121)
print('nu1200 = ', (2*1200*a1+b1)*(16748+750)*1/124)
print('nu1800 = ', (2*1800*a1+b1)*(16748+750)*1/124)

plt.plot(t, T1, 'rx', label='Messwerte')
plt.plot(tlinspace, f(tlinspace, *params), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$t/$s')
plt.ylabel(r'$T_\mathrm{warm}/$K')
plt.axis((-30, 2100, 292, 325))
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/T1.pdf')

plt.clf()


params, covariance_matrix = optimize.curve_fit(f, t, T2)
errors = np.sqrt(np.diag(covariance_matrix))
print('a2 = ', params[0], '+-', errors[0])
print('b2 = ', params[1], '+-', errors[1])
print('c2 = ', params[2], '+-', errors[2])
a2 = ufloat(params[0], errors[0])
b2 = ufloat(params[1], errors[1])
c2 = ufloat(params[2], errors[2])
print('Differential300 = ', 2*300*a2+b2)
print('Differential900 = ', 2*900*a2+b2)
print('Differential1200 = ', 2*1200*a2+b2)
print('Differential1800 = ', 2*1800*a2+b2)

plt.plot(t, T2, 'rx', label='Messwerte')
plt.plot(tlinspace, f(tlinspace, *params), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$t/$s')
plt.ylabel(r'$T_\mathrm{kalt}/$K')
plt.axis((-30, 2100, 275, 295))
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/T2.pdf')

plt.clf()


params, covariance_matrix = optimize.curve_fit(g, 1/T1, np.log(p_b))
errors = np.sqrt(np.diag(covariance_matrix))
print('a3 = ', params[0], '+-', errors[0])
a3runden = ufloat(params[0], errors[0])
print(a3runden)
print('b3 = ', params[1], '+-', errors[1])
a3 = ufloat(params[0], errors[0])
b3 = ufloat(params[1], errors[1])

plt.plot(1/T1, np.log(p_b), 'rx', label='Messwerte')
plt.plot(1/T1, g(1/T1, *params), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$1/T_{\mathrm{warm}}$ in $1/$K')
plt.ylabel(r'$\ln\left(\frac{p_\mathrm{warm}}{p_0}\right)$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/L1.pdf')

plt.clf()

print('Die Verdampfungswärme zum Einsetzen hier ist jetzt')
L = a3runden*1/0.12091*1e-3  # in kJ/kg
print(L)
print('in kJ/kg')

# Massendurchsatz

Qdot = (16748 + 750)*(2*t*a2+b2)
massendurchsatz = Qdot/L
print('Qdot bei t = 300s ist dann ', Qdot[5])
print('Qdot t = 900s ist dann ', Qdot[15])
print('Qdot t = 1200s ist dann ', Qdot[20])
print('Qdot t = 1800s ist dann ', Qdot[30])
print('Der Massendurchsatz bei t = 300s ist dann ', massendurchsatz[5])
print('Der Massendurchsatz bei t = 900s ist dann ', massendurchsatz[15])
print('Der Massendurchsatz bei t = 1200s ist dann ', massendurchsatz[20])
print('Der Massendurchsatz bei t = 1800s ist dann ', massendurchsatz[30])

rho0 = 5.51  # in kg/m^3
T0 = 273.15  # in kelvin
p0 = 1  # in bar
kappa = 1.14  # in gar nix
rhovont = ((rho0*T0)/p0) * p_a/T2
Nmech = 1/(kappa - 1) * (p_b * (p_a/p_b)**(1/kappa) - p_a) * 1/rhovont * massendurchsatz *1e3 *1e-1 # in W, 1e-1 wegen bar nach Pascal

print('array für p_b', p_b)
print('array für p_a', p_a)


print(rhovont)
print(Nmech)

print('rho bei t = 300s ist dann ', rhovont[5])
print('rho bei t = 900s ist dann ', rhovont[15])
print('rho bei t = 1200s ist dann ', rhovont[20])
print('rho bei t = 1800s ist dann ', rhovont[30])
print('Nmech bei t = 300s ist dann ', Nmech[5])
print('Nmech bei t = 900s ist dann ', Nmech[15])
print('Nmech bei t = 1200s ist dann ', Nmech[20])
print('Nmech bei t = 1800s ist dann ', Nmech[30])
