import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


#rührmotoren: 5V
#1  wärmekap. reservoire<. 750 J/K
#V=4l

def f(x, a, b, c):
    return a*x**2+b*x+c


t, T1, T2, p_a, p_b, P = np.genfromtxt('data/data.txt', unpack=True) #t/min T1/°C T2/°C p_a/bar p_b/bar P/Watt

p_a+=1  #Umgebungsdruck
p_b+=1


params, covariance_matrix = optimize.curve_fit(f, t, T1)
errors = np.sqrt(np.diag(covariance_matrix))
print('a1 = ', params[0], '+-', errors[0])
print('b1 = ', params[1], '+-', errors[1])
print('c1 = ', params[2], '+-', errors[2])
a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])
c = ufloat(params[2], errors[2])

plt.plot(t, T1, 'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='fit')
plt.xlabel(r'$t/$min')
plt.ylabel(r'$T_1/$°C')
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
a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])
c = ufloat(params[2], errors[2])

plt.plot(t, T2, 'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='fit')
plt.xlabel(r'$t/$min')
plt.ylabel(r'$T_2/$°C')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/T2.pdf')

plt.clf()
