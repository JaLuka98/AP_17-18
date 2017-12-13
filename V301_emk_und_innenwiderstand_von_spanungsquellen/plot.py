import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat

def f(x, a, b):
    return x*a+b

U_0G=1.335  #Gemessene Leerlaufspannung der Monozelle in V


U, I = np.genfromtxt('data/monozelle.txt', unpack=True) #U/V , I/mA
I*=1e-3 #I/A

params, covariance_matrix = optimize.curve_fit(f, I*1e3, U)

errors = np.sqrt(np.diag(covariance_matrix))

print('a_mono =', params[0]*1e3, '+-', errors[0]*1e3)
print('b_mono =', params[1], '+-', errors[1])
print('Fehler = ', (1.337-U_0G)*100)

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(I*1e3, U, 'rx', label='Messwerte')
plt.plot(I*1e3, f(I*1e3, *params), 'k-', label='fit')
plt.xlabel(r'$I/$mA')
plt.ylabel(r'$U/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/monozelle.pdf')

plt.clf()


U, I = np.genfromtxt('data/gegenspannung.txt', unpack=True) #U/V , I/mA
U2, I2 = np.genfromtxt('data/gegenspannung2.txt', unpack=True) #U/V , I/mA
I*=1e-3 #I/A
I2*=1e-3 #I/A

params, covariance_matrix = optimize.curve_fit(f, I*1e3, U)

errors = np.sqrt(np.diag(covariance_matrix))

print('a_gegen =', params[0]*1e3, '+-', errors[0]*1e3)
print('b_gegen =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

params2, covariance_matrix2 = optimize.curve_fit(f, I2*1e3, U2)

errors2 = np.sqrt(np.diag(covariance_matrix2))

print('a2_gegen =', params2[0]*1e3, '+-', errors2[0]*1e3)
print('b2_gegen =', params2[1], '+-', errors2[1])

a2 = ufloat(params2[0], errors2[0])
b2 = ufloat(params2[1], errors2[1])

plt.plot(I*1e3, U, 'rx', label='Messwerte')
plt.plot(I*1e3, f(I*1e3, *params), 'k-', label='fit vor dem Sprung')
plt.plot(I2*1e3, U2, 'rx')
plt.plot(I2*1e3, f(I2*1e3, *params2), 'b-', label='fit nach dem Sprung')
plt.xlabel(r'$I/$mA')
plt.ylabel(r'$U/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/gegenspannung.pdf')

plt.clf()


U, I = np.genfromtxt('data/rechteck.txt', unpack=True) #U/V , I/mA
I*=1e-3 #I/A

params, covariance_matrix = optimize.curve_fit(f, I*1e3, U)

errors = np.sqrt(np.diag(covariance_matrix))

print('a_rechteck =', params[0]*1e3, '+-', errors[0]*1e3)
print('b_rechteck =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(I*1e3, U, 'rx', label='Messwerte')
plt.plot(I*1e3, f(I*1e3, *params), 'k-', label='fit')
plt.xlabel(r'$I/$mA')
plt.ylabel(r'$U/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/rechteck.pdf')

plt.clf()


U, I = np.genfromtxt('data/sinus.txt', unpack=True) #U/V , I/mA
I*=1e-3 #I/A

params, covariance_matrix = optimize.curve_fit(f, I*1e3, U)

errors = np.sqrt(np.diag(covariance_matrix))

print('a_sinus =', params[0]*1e3, '+-', errors[0]*1e3)
print('b_sinus =', params[1], '+-', errors[1])

a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(I*1e3, U, 'rx', label='Messwerte')
plt.plot(I*1e3, f(I*1e3, *params), 'k-', label='fit')
plt.xlabel(r'$I/$mA')
plt.ylabel(r'$U/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/sinus.pdf')

plt.clf()


U, I = np.genfromtxt('data/monozelle.txt', unpack=True) #U/V , I/mA
I*=1e-3 #I/A

N=U*I
R=U/I

U0=1.337
Ri=6.446

l=np.linspace(0, 60, 1000)

print('U*I', N*1e3)
print('U/I', R)

plt.plot(R, N, 'rx', label='Messwerte')
plt.plot(l, (U0**2*l)/((l+Ri)**2), 'b-', label='Theoriekurve')
plt.xlabel(r'$R_a/$Ohm')
plt.ylabel(r'$N/$W')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/leistung.pdf')

plt.clf()
