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
d_1=2.2 #cm
d_2=1.8 #cm

d_1*=1e-2    #in meter
d_2*=1e-2    #in meter


#Erste Messung
print('Erste Messung:')
p, channel, N_ges  = np.genfromtxt('data/druck.txt', unpack=True)
p*=1e-3 #p in bar

x=d_1*p/p_0
Z=N_ges/t

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm']
m = np.zeros((21, 4))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
table=matrix2latex(m, headerRow=hr, format='%.2f')
print(table)




#Zweite Messung
print('Zweite Messung:')
p, channel, N_ges  = np.genfromtxt('data/druck_2.txt', unpack=True)
p*=1e-3
x=d_1*p/p_0
Z=N_ges/t

hr = ['p/mbar','channel','$N_{ges}$', '$x$/cm']
m = np.zeros((21, 4))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
table=matrix2latex(m, headerRow=hr, format='%.2f')
print(table)

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
