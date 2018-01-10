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

def g(x, a, b):
    return x*a+b



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
a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])
c = ufloat(params[2], errors[2])

plt.plot(t, T1, 'rx', label='Messwerte')
plt.plot(t, f(t, *params), 'k-', label='fit')
plt.xlabel(r'$t/$s')
plt.ylabel(r'$T_1/$K')
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
plt.xlabel(r'$t/$s')
plt.ylabel(r'$T_2/$K')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/T2.pdf')

plt.clf()


params, covariance_matrix = optimize.curve_fit(g, 1/T1, np.log(p_b))
errors = np.sqrt(np.diag(covariance_matrix))
print('a3 = ', params[0], '+-', errors[0])
print('b3 = ', params[1], '+-', errors[1])
a = ufloat(params[0], errors[0])
b = ufloat(params[1], errors[1])

plt.plot(1/T1, np.log(p_b), 'rx', label='Messwerte')
plt.plot(1/T1, g(1/T1, *params), 'k-', label='fit')
plt.xlabel(r'$1/T_1$ in $1/$K')
plt.ylabel(r'$\ln(\frac{p_b}{p_0}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/L1.pdf')

plt.clf()
