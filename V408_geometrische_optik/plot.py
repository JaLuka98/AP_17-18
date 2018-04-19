import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import correlated_values
from uncertainties import ufloat
from matrix2latex import matrix2latex


def linearfit(x, a, b):
    return a*x+b


f_1 = 150  # Brennweite der Bekannten Linse in mm; benutzt für erste Messreihe und Bessel
f_2 = -50   # Brennweite einer Linse bei Abbe in mm
f_3 = 50   # Brennweite einer Linse bei Abbe in mm#

G = 3  # Gegenstandsgröße in cm

g, b, B = np.genfromtxt('data/bekannt.txt', unpack=True)  # in cm
V1 = B/G
V2 = b/g
f = np.genfromtxt('data/f.txt', unpack=True) #in cm
#Kannst du noch f oder so hier rein tun?
hr = ['$g/$cm', '$b/$cm', '$B/$cm', '$V1/$cm', '$V2/$cm', '$f/$cm']
m = np.zeros((11, 6))
m[:,0] = g
m[:,1] = b
m[:,2] = B
m[:,3] = V1
m[:,4] = V2
m[:,5] = f
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

# Diesen komischen Plot erzeugen
for i in range(0, len(b), 1):
    plt.plot([0, g[i]], [b[i], 0], 'rx-')

einsdurchf = 1/g + 1/b
f = 1/einsdurchf
np.savetxt('f.txt', np.column_stack([f]))

# Ausreißer entfernen
g = np.delete(g, np.argwhere(g == 19.4))
g = np.delete(g, np.argwhere(g == 56.0))
b = np.delete(b, np.argwhere(b == 11.4))
b = np.delete(b, np.argwhere(b == 35.6))
print(g)
print(b)
einsdurchf = 1/g + 1/b
f = 1/einsdurchf
print(f)
f = ufloat(np.mean(f), np.std(f, ddof=1))
print('f für Linse mit f=150cm mit herausgerechneten Ausreissern ist f=', f, 'cm')


plt.xlabel(r'$g/$cm')
plt.ylabel(r'$b/$cm')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('build/bekanntgross.pdf')
plt.axis([12, 17, 12, 17])
plt.savefig('build/bekanntklein.pdf')

plt.clf()

g, b = np.genfromtxt('data/wasser.txt', unpack=True)  # in cm
f = np.genfromtxt('data/f_wasser.txt', unpack=True) #in cm

for i in range(0, len(b), 1):
    plt.plot([0, g[i]], [b[i], 0], 'rx-')

hr = ['$g/$cm', '$b/$cm', '$f/$cm']
m = np.zeros((10, 3))
m[:,0] = g
m[:,1] = b
m[:,2] = f
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)


einsdurchf = 1/g + 1/b
f = 1/einsdurchf
print(f)
np.savetxt('f_wasser.txt', np.column_stack([f]))
f = ufloat(np.mean(f), np.std(f, ddof=1))
print('f für Wasserlinse mit unbekannter Brennweite ist f=', f, 'cm')

plt.xlabel(r'$g/$cm')
plt.ylabel(r'$b/$cm')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.savefig('build/wassergross.pdf')
plt.axis([10, 15, 10, 15])
plt.savefig('build/wasserklein.pdf')

plt.clf()

# Methode von Bessel

# Weisses Licht
e, g1, b1, g2, b2 = np.genfromtxt('data/bessel.txt', unpack=True)  # in cm
d1 = np.abs(g1-b1)
d2 = np.abs(g2-b2)
d = (d1+d2)/2
f = (e**2-d**2)/(4*e)

hr = ['$e/$cm', '$g_1/$cm', 'b_1$/$cm', '$g_2/$cm', '$b_2/$cm', '$f/$cm']
m = np.zeros((10, 6))
m[:,0] = e
m[:,1] = g1
m[:,2] = b1
m[:,3] = g2
m[:,4] = b2
m[:,5] = f
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

f = ufloat(np.mean(f), np.std(f, ddof=1))
print('Die Brennweite der Linse (weisses Licht) mit f=150mm mit der Methode von Bessel ist f=', f, 'cm')


# Blaues Licht

e, g1, b1, g2, b2 = np.genfromtxt('data/bessel_blau.txt', unpack=True)  # in cm
d1 = np.abs(g1-b1)
d2 = np.abs(g2-b2)
d = (d1+d2)/2
f = (e**2-d**2)/(4*e)

hr = ['$e/$cm', '$g_1/$cm', 'b_1$/$cm', '$g_2/$cm', '$b_2/$cm', '$f/$cm']
m = np.zeros((5, 6))
m[:,0] = e
m[:,1] = g1
m[:,2] = b1
m[:,3] = g2
m[:,4] = b2
m[:,5] = f
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

f = ufloat(np.mean(f), np.std(f, ddof=1))
print('Die Brennweite der Linse (blaues Licht) mit f=150mm mit der Methode von Bessel ist f=', f, 'cm')

# Rotes Licht

e, g1, b1, g2, b2 = np.genfromtxt('data/bessel_rot.txt', unpack=True)  # in cm
d1 = np.abs(g1-b1)
d2 = np.abs(g2-b2)
d = (d1+d2)/2
f = (e**2-d**2)/(4*e)

hr = ['$e/$cm', '$g_1/$cm', 'b_1$/$cm', '$g_2/$cm', '$b_2/$cm', '$f/$cm']
m = np.zeros((5, 6))
m[:,0] = e
m[:,1] = g1
m[:,2] = b1
m[:,3] = g2
m[:,4] = b2
m[:,5] = f
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

f = ufloat(np.mean(f), np.std(f, ddof=1))
print('Die Brennweite der Linse (rotes Licht) mit f=150mm mit der Methode von Bessel ist f=', f, 'cm')

# Und die Methode von Abbe folgt zuletzt
d=60 #m
durchftheo= 1/(50)-1/(50) + d/(50**2)
ftheo=1/durchftheo
print('Der Theoriewert für das Linsensystem beträgt f_theo=', ftheo, 'mm')


gstrich, bstrich, B = np.genfromtxt('data/abbe.txt', unpack=True)  # in cm
V = B/G
print('V =', V)
print('1+1/V', 1+1/V)

hr = ['$g_strich/$cm', '$b_strich$cm', '$B/$cm', '$V$']
m = np.zeros((10, 4))
m[:,0] = gstrich
m[:,1] = bstrich
m[:,2] = B
m[:,3] = V
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

params, covariance_matrix = optimize.curve_fit(linearfit, 1+1/V, gstrich)

fg, h = correlated_values(params, covariance_matrix)

print('Die Abbe-Methode mit gstrich')
print('fg =', fg)
print('h =', h)

plt.plot(1+1/V, gstrich, 'rx', label='Messwerte')

xlinspace = np.linspace(2.2, 8.1, 500)
plt.plot(xlinspace, linearfit(xlinspace, *params), 'k-', label='Ausgleichsfunktion')

plt.xlabel(r'$1+\frac{1}{V}$')
plt.ylabel(r'$g^\prime/$cm')
plt.tight_layout()
plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
plt.axis([2.3, 8, 10, 37])
plt.savefig('build/abbegstrich.pdf')
plt.clf()

params, covariance_matrix = optimize.curve_fit(linearfit, 1+V, bstrich)

fb, hstrich = correlated_values(params, covariance_matrix)

print('Die Abbe-Methode mit bstrich')
print('fb =', fb)
print('hstrich =', hstrich)
print('Mittelwert von fg und fb', (fg+fb)/2)

plt.plot(1+V, bstrich, 'rx', label='Messwerte')

xlinspace = np.linspace(1, 1.8, 500)
plt.plot(xlinspace, linearfit(xlinspace, *params), 'k-', label='Ausgleichsfunktion')

plt.xlabel(r'$1+V$')
plt.ylabel(r'$b^\prime/$cm')
plt.tight_layout()
plt.legend()
plt.grid()
plt.autoscale(enable=True, axis='x', tight=True)
#plt.axis([2.3, 8, 10, 37])
plt.savefig('build/abbebstrich.pdf')
plt.clf()
