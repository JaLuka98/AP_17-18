import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


f_1=150 #Brennweite der Bekannten Linse in mm; benutzt für erste Messreihe und Bessel
f_2=-50  #Brennweite einer Linse bei Abbe in mm
f_3=50  #Brennweite einer Linse bei Abbe in mm#

G=3 #Gegenstandsgröße in cm

g, b, B = np.genfromtxt('data/bekannt.txt', unpack=True) # in cm

for i in range(0, len(b), 1):
    plt.plot([0, g[i]], [b[i], 0], 'rx-')

plt.xlabel(r'$g/$cm')
plt.ylabel(r'$b/$cm')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('build/bekannt.pdf')

plt.clf()

g, b = np.genfromtxt('data/wasser.txt', unpack=True) # in cm

for i in range(0, len(b), 1):
    plt.plot([0, g[i]], [b[i], 0], 'rx-')

einsdurchf = 1/g + 1/b
f = 1/einsdurchf
print(f)

plt.xlabel(r'$g/$cm')
plt.ylabel(r'$b/$cm')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('build/wasser.pdf')

plt.clf()
