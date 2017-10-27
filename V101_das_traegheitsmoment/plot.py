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
r += 0.0298/2  # weil die "Punktmassen" Zylinder sind, hzyl = 2.98cm
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
plt.xlabel('$r²$/cm²')
plt.ylabel('$T²$/s²')
plt.legend()
plt.grid()

plt.savefig('build/drillachse.pdf')

# Erstellen einer Tabelle für die latex-Datei
np.savetxt('data/tabelledrillachse.txt', np.transpose([r*1e2, T, (r*1e2)**2, T**2]), fmt='%1.2f')

I_D = (b*8*(np.pi)**2*m/a)/(4*(np.pi)**2)*1e3
print('Das Eigentraegheitsmoment I_D der Drillachse berechnet sich zu', I_D)

dreiT = np.genfromtxt('data/zylinder.txt', unpack=True)

T = dreiT/3
print('Die Periodendauern sind', T)
T = ufloat(np.sum(T)/len(T), np.std(T))
I_zylinder = (T**2*D)/(4*(np.pi)**2)*1e3
print('Der experimentelle Wert fuer das Traegheitsmoment des Zylinders ist', I_zylinder-I_D)

m_z = 2.3959
r_z = 0.05
print('Der theoretische Wert fuer das Traegheitsmoment des Zylinders ist', (m_z*r_z**2/2)*1e3)

dreiT = np.genfromtxt('data/kugel.txt', unpack=True)

T = dreiT/3
T = ufloat(np.sum(T)/len(T), np.std(T))
I_Kugel = (T**2*D)/(4*(np.pi)**2)*1e3
print('Der experimentelle Wert fuer das Traegheitsmoment der Kugel ist', I_Kugel)

m_k = 0.8125
d_kugel = np.array([0.1374, 0.1372, 0.1373, 0.1373, 0.1370])
d_kugel = ufloat(np.sum(d_kugel)/len(d_kugel), np.std(d_kugel))
print('Der theoretische Wert fuer das Traegheitsmoment der Kugel ist', (0.4*m_k*(d_kugel/2)**2)*1e3)

# Berechnung der Puppe
# 1. Körperhaltung war angelegte Arme und Spagat (zur Seite)
# 2 Körperhaltung war zur Seite abgespreizte Arme und ein Spagat nach vorne und hinten

# Maße der Puppe
d_kopf = 3.1  # in cm
h_arm = 14  # in cm
d_arm = 1.43  # in cm
d_rumpf = 3.60  # in cm
h_rumpf = 9.76  # in cm
h_bein = 15.38  # in cm
d_bein = 1.68  # in cm

rho_ahorn=650

I_K=9.744e-7
I_R=1.046e-5
I_A1=3.736e-7
I_A2=2.406e-5
I_B=4.407e-5

I_Puppe1=13.193e-3
I_Puppe2=27.463e-3
