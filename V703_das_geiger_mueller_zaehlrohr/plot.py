import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex

def f(x, a, b):
   return a*x + b

t=60 #s

#Zählrohr charakteristik
print('Bestimmung der Zählrohrcharakteristik:')

U, N, I = np.genfromtxt('data/charakteristik.txt', unpack=True) #U/V und I/µA
I*=1e-6 #I/A

A = (N/t)
sigmaA = np.sqrt(N)/t

params, covariance_matrix = optimize.curve_fit(f, U[2:], A[2:])
a, b = correlated_values(params, covariance_matrix)
print('a =', a)
print('b =', b)

linspace = np.linspace(290, 710, 1000)
plt.errorbar(U[0:2], A[0:2], yerr= np.sqrt(N[0:2])/t, fmt = 'gx', mew = 0.5,
             capsize=2, ecolor='b', elinewidth=0.5, markeredgewidth=0.5, label='ausgenommene Messwerte')
plt.errorbar(U[2:], A[2:],yerr= np.sqrt(N[2:])/t,fmt = 'rx', mew = 0.5,
             capsize=2, ecolor='b', elinewidth=0.5, markeredgewidth=0.5, label='Messwerte')
plt.plot(linspace, f(linspace, *params), 'm-',linewidth=0.5, label='Ausgleichsfunktion')
plt.axvline(320 , color='g', linestyle='--', linewidth=0.5)
plt.axvline(700, color='g', linestyle='--', linewidth=0.5)
plt.xlabel(r'$U/$V')
plt.ylabel(r'$Z/$(1/s)')
plt.xlim(290,710)
plt.tight_layout()
plt.legend()
plt.grid()
plt.tight_layout()
plt.grid()
plt.savefig('build/charakteristik.pdf')
plt.clf()


#Freigesetzte Ladung
print('Bestimmung der freigesetzen Ladungen:')

#N_f=ufloat(N[1:], unp.sqrt(N[1:])) #Hier muss auch noch der Fehler vom N beachtet werden!


#N_f=np.empty([41])
#for i in range(0, 41):
#    N_f[i]=ufloat(N[i], unp.sqrt(N[i]))

Q=I*t/N               #Ladung in Coulomb
Q_e=Q/(1.602*1e-19)      #Anzahl der Elektronen
Q_eerror=np.sqrt(Q_e)
print(Q_eerror)

#print('Ladung in Coulomb: Q=', Q)
#print('Anzahl der Elektronen: N=',Q_e)

plt.plot(U, Q_e, 'rx', mew=0.5, label='Messwerte')
#plt.errorbar(U, Q_e, yerr=Q_eerror, fmt = 'rx', mew = 0.5,markersize=2,
#capsize=2, ecolor='b', elinewidth=0.5, markeredgewidth=0.5, label='Messwerte')
plt.xlabel(r'$U/$V')
plt.ylabel(r'$(\Delta Q/e_0)$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/ladung.pdf')
plt.clf()


hr = ['$U/$V', '$N$', '$sigmaN$','$A/\frac{1}{\text{s}}$', 'simgaA', '$I$µA', '$\Delta Q/e_0 \cdot \symup{10^{10}}$']
m = np.zeros((41, 8))
m[:, 0] = U
m[:, 1] = N
m[:, 2] = np.sqrt(N)
m[:, 3] = A
m[:, 4] = sigmaA
m[:, 5] = I*1e6
m[:, 6] = Q_e*1e-10
m[:, 7] = Q_eerror*1e-10
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)

#Bestimmung der Totzeiten
print('Bestimmung der Totzeiten:')

N_1=12274
N_2=13075
N_12=24705

A1=N_1/t
A2=N_2/t
A12=N_12/t

A_1=ufloat(A1, unp.sqrt(N_1)/t)
A_2=ufloat(A2, unp.sqrt(N_2)/t)
A_12=ufloat(A12, unp.sqrt(N_12)/t)
print('A_1=', A_1)
print('A_2=', A_2)
print('A_12=', A_12)

U, T, T_e = np.genfromtxt('data/totzeit.txt', unpack=True) #U/V, T/µs, I/µA
T*=1e-6 #T/s
T_e*=1e-6 #T/s

T_mittel = ufloat(np.mean(T), np.std(T))
print('Der Mittelwert der Totzeit T ist:', T_mittel)


T_exp = (A_1+A_2-A_12)/(2*A_1*A_2)
print('Die Totzeit über die Proben ist:', T_exp)
