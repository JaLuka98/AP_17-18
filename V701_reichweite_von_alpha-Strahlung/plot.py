import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex
from uncertainties import correlated_values
from scipy import stats

t=120 #s
p_0=1.013 #bar
d_1=2.2 #cm
d_2=1.8 #cm

d_1*=1e-2    #in meter
d_2*=1e-2    #in meter

def f(x, a, b):
   return a*x + b



#Erste Messung
print('Erste Messung:')
p, channel, N_ges  = np.genfromtxt('data/druck.txt', unpack=True)
p*=1e-3 #p in bar

x=d_1*p/p_0
Z=N_ges/t

params, covariance_matrix = optimize.curve_fit(f, x[16:20], Z[16:20])
a, b = correlated_values(params, covariance_matrix)
print('a =', a)
print('b =', b)
linspace = np.linspace(0.0, 0.03, 1000)

plt.plot(x, Z, 'rx', mew=0.5, label='Messwerte')
plt.plot(x[16:20], Z[16:20], 'bx', mew=0.5, label='Für die Ausgleichsrechnung verwendete Messwerte')
plt.plot(linspace, f(linspace, *params), 'm-',linewidth=0.5, label='Ausgleichsfunktion')
plt.xlabel(r'$x/$m')
plt.ylabel(r'$Z/(1/s)$')
plt.xlim(-0.001,0.023)
plt.ylim(150, 800)
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/druck1.pdf')
plt.clf()

R_m=(Z[0]*1/2-b)/a
print('Mittlere Reichweite:', R_m)
E_m=(10/31)**(2/3)*(R_m*1e3)**(2/3)
print('Mittlere Energie:', E_m)

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm','Z']
m = np.zeros((21, 5))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
m[:,4] = Z
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)



#jetzt die Energie
print('Jetzt die Energie:')

E=channel*4/966

params, covariance_matrix = optimize.curve_fit(f, x[:18], E[:18])
a, b = correlated_values(params, covariance_matrix)
print('a =', a)
print('b =', b)
linspace = np.linspace(-0.1, 0.03, 1000)

plt.plot(x, E, 'rx', mew=0.5, label='Messwerte')
plt.plot(x[:18], E[:18], 'bx', mew=0.5, label='Für die Ausgleichsrechnung verwendete Messwerte')
plt.plot(linspace, f(linspace, *params), 'm-',linewidth=0.5, label='Ausgleichsfunktion')
plt.xlabel(r'$x/$m')
plt.ylabel(r'$E/$MeV')
plt.xlim(-0.001,0.023)
plt.ylim(2.1, 4.1)
plt.tight_layout()
plt.legend(prop={'size': 9})
plt.grid()
plt.savefig('build/energie1.pdf')
plt.clf()


#Zweite Messung
print('Zweite Messung:')

p, channel, N_ges  = np.genfromtxt('data/druck_2.txt', unpack=True)
p*=1e-3

x=d_2*p/p_0
Z=(N_ges)/120  #mit t wollte er es irgendwie nicht... warum auch immer...

params, covariance_matrix = optimize.curve_fit(f, x[14:19], Z[14:19])
a, b = correlated_values(params, covariance_matrix)
print('a =', a)
print('b =', b)
linspace = np.linspace(0.0, 0.03, 1000)

plt.plot(x, Z, 'rx', mew=0.5, label='Messwerte')
plt.plot(x[14:19], Z[14:19], 'bx', mew=0.5, label='Für die Ausgleichsrechnung verwendete Messwerte')
plt.plot(linspace, f(linspace, *params), 'm-',linewidth=0.5, label='Ausgleichsfunktion')
plt.xlabel(r'$x/$m')
plt.ylabel(r'$Z/(1/s)$')
plt.xlim(-0.001,0.023)
plt.ylim(750, 1020)
plt.tight_layout()
plt.legend(prop={'size': 9})
plt.grid()
plt.savefig('build/druck2.pdf')
plt.clf()

R_m=(Z[0]*1/2-b)/a
print('Mittlere Reichweite:', R_m)
E_m=(10/31)**(2/3)*(R_m*1e3)**(2/3)
print('Mittlere Energie:', E_m)

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm','Z']
m = np.zeros((21, 5))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
m[:,4] = Z
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)




#jetzt die Energie
print('Jetzt die Energie:')

E2=channel*4/1007

params, covariance_matrix = optimize.curve_fit(f, x[1:20], E2[1:20])
a, b = correlated_values(params, covariance_matrix)
print('a =', a)
print('b =', b)
linspace = np.linspace(-0.1, 0.03, 1000)

plt.plot(x, E2, 'rx', mew=0.5, label='Messwerte')
plt.plot(x[1:20], E2[1:20], 'bx', mew=0.5, label='Für die Ausgleichsrechnung verwendete Messwerte')
plt.plot(linspace, f(linspace, *params), 'm-',linewidth=0.5, label='Ausgleichsfunktion')
plt.xlabel(r'$x/$m')
plt.ylabel(r'$E/$MeV')
plt.xlim(-0.001,0.023)
plt.ylim(2.3, 4.4)
plt.tight_layout()
plt.legend(prop={'size': 9})
plt.grid()
plt.savefig('build/energie2.pdf')
plt.clf()

hr = ['$p$/mbar','$E_1$/MeV','$E_2$/MeV']
m = np.zeros((21, 3))
m[:,0] = p*1e3
m[:,1] = E
m[:,2] = E2

t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

# Histamin
N  = np.genfromtxt('data/statistik.txt', unpack=True)
N /= 10
#plt.hist(N, bins=22, normed = True)
#plt.savefig('build/hist.pdf')
print('Mittelwert der Stichprobe', np.mean(N))
print(stats.sem(N))
print('Varianz der Stichprobe', np.var(N, ddof = 1))
print('Standardabweichung der Stichprobe', np.std(N, ddof = 1))

g = np.random.normal(np.mean(N), np.std(N), 10000)
#p = np.random.poisson(np.mean(N), 10000)
plt.hist([N,g], 20, label=['Messwert','Gaußverteilung'], alpha=1, normed=1)
plt.legend()
plt.ylabel(r'Normierte Häufigkeit')
plt.xlabel(r'Z / $\frac{1}{s}$')
plt.gcf().subplots_adjust(bottom=0.18)
plt.grid()
plt.savefig('build/hist.pdf')
