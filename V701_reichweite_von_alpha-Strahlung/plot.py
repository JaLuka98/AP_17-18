import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex
from scipy import stats

t=120 #s
p_0=1.013 #bar
d1=2.2 #cm
d2=1.8 #cm

d_1*=1e-2    #in meter
d_2*=1e-2    #in meter


#Erste Messung
print('Erste Messung:')
p, channel, N_ges  = np.genfromtxt('data/druck.txt', unpack=True)
p*=1e-3 #p in bar

x=d_1*p/p_0
Z=N_ges/t

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm','Z']
m = np.zeros((21, 5))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
m[:,4] = Z
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

plt.plot(x*1e2, Z, 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$x/$cm')
plt.ylabel(r'$(Z/(1/s))$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/druck1.pdf')
plt.clf()





#jetzt die Energie

E=channel*4/966

plt.plot(x[:18]*1e2, E[:18], 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$x/$cm')
plt.ylabel(r'$(E/$MeV')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/energie1.pdf')
plt.clf()


#Zweite Messung
print('Zweite Messung:')

p, channel, N_ges  = np.genfromtxt('data/druck_2.txt', unpack=True)
p*=1e-3

x=d2*p/p_0
Z=(N_ges)/120  #mit t wollte er es irgendwie nicht... warum auch immer...

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm','Z']
m = np.zeros((21, 5))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
m[:,4] = Z
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

plt.plot(x*1e2, Z, 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$x/$cm')
plt.ylabel(r'$(Z/(1/s))$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/druck2.pdf')
plt.clf()


#jetzt die Energie

E=channel*4/1007

plt.plot(x*1e2, E, 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$x/$cm')
plt.ylabel(r'$(E/$MeV')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/energie2.pdf')
plt.clf()

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
plt.xlabel(r'N / $\frac{1}{s}$')
plt.gcf().subplots_adjust(bottom=0.18)
plt.grid()
plt.savefig('build/hist.pdf')
