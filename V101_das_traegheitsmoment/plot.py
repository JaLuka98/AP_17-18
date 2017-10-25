import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def f(x, a, b):
    return x*a+b


# Bestimmung der winkelrichtgroesse
phi, r, F = np.genfromtxt('data/winkelrichtgroesse.txt', unpack=True)
# phi in grad, r in cm, F in N
phi *= 2*np.pi/360  # phi in rad
r *= 1e-2  # r in m

Darray = F * r / phi

D = ufloat(np.sum(Darray)/len(Darray), np.std(Darray))
print('Die winkelrichtgroesse bei der statischen Methode berechnet sich zu', D)

# Bestimmung des Eigenträgheitsmoments der drillachse
r, zweiT = np.genfromtxt('data/drillachse.txt', unpack=True)
# a in cm, 2T ins
r *= 1e-2  # r in m
r += 0.0298  # weil die "Punktmassen" Zylinder sind, hzyl = 2.98cm
T = zweiT/2
m1 = 0.2324
m2 = 0.2225
m = (m1+m2)/2  # m gemittelt

params, covariance_matrix = optimize.curve_fit(f, r**2, T**2)

errors = np.sqrt(np.diag(covariance_matrix))

print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(r**2, T**2, 'rx', label='Messwerte')
plt.plot(r**2, f(r**2, *params), 'k-', label='fit')
plt.xlabel('$a²$/cm²')
plt.ylabel('$T²$/s²')

plt.savefig('build/drillachse.pdf')

I_D = (b*8*(np.pi)**2*m/a)/(4*(np.pi)**2)
print('Das Eigentraegheitsmoment I_D der Drillachse berechnet sich zu', I_D)

dreiT = np.genfromtxt('data/zylinder.txt', unpack=True)

T = dreiT/3
T = ufloat(np.sum(T)/len(T), np.std(T))
I_zylinder = (T**2*D)/(4*(np.pi)**2)
print('Der experimentelle Wert fuer das Traegheitsmoment des Zylinders ist', I_zylinder)

m_z = 2.3959
r_z = 0.05
print('Der theoretische Wert fuer das Traegheitsmoment des Zylinders ist', m_z*r_z**2/2)

dreiT = np.genfromtxt('data/kugel.txt', unpack=True)

T = dreiT/3
T = ufloat(np.sum(T)/len(T), np.std(T))
I_Kugel = (T**2*D)/(4*(np.pi)**2)
print('Der experimentelle Wert fuer das Traegheitsmoment der Kugel ist', I_Kugel)

m_k = 0.8125
d_kugel = np.array([0.1374, 0.1372, 0.1373, 0.1373, 0.1370])
d_kugel = ufloat(np.sum(d_kugel)/len(d_kugel), np.std(d_kugel))
print('Der theoretische Wert fuer das Traegheitsmoment der Kugel ist', 0.4*m_k*(d_kugel/2)**2)
