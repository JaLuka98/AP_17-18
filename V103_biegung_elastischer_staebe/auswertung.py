import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def f(x, a, b):
    return a*x+b


# Anfang einseitige Einspannung und runder Querschnitt
x1, D1, D2 = np.genfromtxt('rundeinseitig.txt', unpack=True)
# in cm, mm, mm
m = 0.7884  # in kg
L = 0.54  # in m
x1 *= 1e-2  # x1 jetzt in m
d = 8.1  # Durchmesser d des Stabs in mm
d *= 1e-3  # d jetzt in m
I = (np.pi/4)*((d/2)**4)  # Flaechentraegheitsmoment für runden Querschnitt  in m^4
D = D2-D1  # Die Ausgleichsrechnung, um die Biegung in Nullage zu beachten
xwerte = L*(x1**2)-(x1**3)/3  # Die Gleichung für einseitige Einspannung
xwerte *= 1e3  # um auf der x-Achse mm zu haben

params, covariance_matrix = optimize.curve_fit(f, xwerte, D)

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])

print('E/10^-9 fuer rund, einseitige Einspannung ist', (m*9.81)/(2*I*a)*1e-9)

plt.plot(xwerte, D, 'rx', label='Messwerte')
plt.plot(xwerte, f(xwerte, *params), 'b-', label='fit')
plt.title('Runder Stab, einseitige Einspannung')
plt.legend()
plt.grid()
plt.xlabel(r'$(Lx^2-\frac{x^3}{3})/$mm³')
plt.ylabel(r'$D(x)/$mm')
plt.savefig('build/rundeinseitig.pdf')
# Ende einseitige Einspannung und runder Querschnitt


plt.clf()


# Anfang einseitige Einspannung und quadratischer Querschnitt
x1, D1, D2 = np.genfromtxt('quadratischeinseitig.txt', unpack=True)
# in cm, mm, mm
m = 2.3682  # in kg
L = 0.54  # in m
x1 *= 1e-2  # x1 jetzt in m
a = 8.0  # Seitenlaenge a in mm
a *= 1e-3  # a jetzt in m
I = (a**4)/12  # Flaechentraegheitsmoment für einen quadratischen Querschnitt in m^4
D = D2-D1  # Die Ausgleichsrechnung, um die Biegung in Nullage zu beachten
xwerte = L*(x1**2)-(x1**3)/3  # Die Gleichung für einseitige Einspannung
xwerte *= 1e3  # um auf der x-Achse mm zu haben

params, covariance_matrix = optimize.curve_fit(f, xwerte, D)

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])

print('E/10^-9 fuer eckig, einseitige Einspannung ist', (m*9.81)/(2*I*a)*1e-9)

plt.plot(xwerte, D, 'rx', label='Messwerte')
plt.plot(xwerte, f(xwerte, *params), 'b-', label='fit')
plt.title('Eckiger Stab, einseitige Einspannung')
plt.legend()
plt.grid()
plt.xlabel(r'$(Lx^2-\frac{x^3}{3})/$mm³')
plt.ylabel(r'$D(x)/$mm')
plt.savefig('build/quadratischeinseitig.pdf')
# Ende einseitige Einspannung und quadratischer Querschnitt


plt.clf()


# Anfang beidseitige Einspannung und quadratischer Querschnitt
x1, D1, D2 = np.genfromtxt('quadratischzweiseitig.txt', unpack=True)
# in cm, mm, mm
m = 2.3682  # in kg
L = 0.54  # in m
x1 *= 1e-2  # x1 jetzt in m
a = 8.0  # Seitenlaenge a in mm
a *= 1e-3  # a jetzt in m
I = (a**4)/12  # Flaechentraegheitsmoment für einen quadratischen Querschnitt in m^4
D = D2-D1  # Die Ausgleichsrechnung, um die Biegung in Nullage zu beachten
xwerte = 3*(L**2)*x1 - 4*x1**3  # Die Gleichung für beidseitige Einspannung
xwerte *= 1e3

params, covariance_matrix = optimize.curve_fit(f, xwerte, D)

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
print(D, xwerte)

print('E/10^-9 fuer eckig, zweiseitige Einspannung ist', (m*9.81)/(48*I*a)*1e-9)

plt.plot(xwerte, D, 'rx', label='Messwerte')
plt.plot(xwerte, f(xwerte, *params), 'b-', label='fit')
plt.legend()
plt.grid()
plt.title('Eckiger Stab, beidseitige Einspannung')
plt.xlabel(r'$(3L^2x-4x^3)/$mm³')
plt.ylabel(r'$D(x)/$mm')
plt.savefig('build/quadratischzweiseitig.pdf')
