import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex


def linearfit(x, a, b):
    return a*x+b


U_g = unp.uarray(np.zeros(6), np.zeros(6))  # brauchen wir später
gelbU, gelbI = np.genfromtxt('data/gelb.txt', unpack=True)
gruenU, gruenI = np.genfromtxt('data/gruen.txt', unpack=True)
gruenblauU, gruenblauI = np.genfromtxt('data/gruenblau.txt', unpack=True)
violett1U, violett1I = np.genfromtxt('data/violett1.txt', unpack=True)
violett2U, violett2I = np.genfromtxt('data/violett2.txt', unpack=True)
ultraviolettU, ultraviolettI = np.genfromtxt('data/ultraviolett.txt', unpack=True)

# Die gelbe (orangefarbene) Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gelbU, np.sqrt(gelbI))
a, b = correlated_values(params, covariance_matrix)
U_g[0] = -b/a
print('U_g für gelb =', -b/a)
gelblinspace = np.linspace(-0.1, 0.7, 500)

plt.plot(gelbU, np.sqrt(gelbI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gelblinspace, linearfit(gelblinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.6, -1, 11])
plt.savefig('build/gelb.pdf')
plt.clf()

#hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
#m = np.zeros((np.size(gelbU), 3))
#m[:,0] = gelbU
#m[:,1] = gelbI
#m[:,2] = np.sqrt(gelbI)
#t = matrix2latex(m, headerRow=hr, format='%.2f')
#print(t)

# Die gruene Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gruenU, np.sqrt(gruenI))
a, b = correlated_values(params, covariance_matrix)
U_g[1] = -b/a
print('U_g für grün =', -b/a)
gruenlinspace = np.linspace(-0.1, 0.7, 500)

plt.plot(gruenU, np.sqrt(gruenI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gruenlinspace, linearfit(gruenlinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.9, -1, 30])
plt.savefig('build/gruen.pdf')
plt.clf()

#hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
#m = np.zeros((np.size(gruenU), 3))
#m[:,0] = gruenU
#m[:,1] = gruenI
#m[:,2] = np.sqrt(gruenI)
#t = matrix2latex(m, headerRow=hr, format='%.2f')
#print(t)

# Die gruenblaue Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, gruenblauU, np.sqrt(gruenblauI))
a, b = correlated_values(params, covariance_matrix)
U_g[2] = -b/a
print('U_g für gruenblau =', -b/a)
gruenblaulinspace = np.linspace(-0.1, 0.9, 500)

plt.plot(gruenblauU, np.sqrt(gruenblauI), 'rx', label='Messwerte', mew=0.5)
plt.plot(gruenblaulinspace, linearfit(gruenblaulinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 0.90, -1, 6])
plt.savefig('build/gruenblau.pdf')
plt.clf()

#hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
#m = np.zeros((np.size(gruenblauU), 3))
#m[:,0] = gruenblauU
#m[:,1] = gruenblauI
#m[:,2] = np.sqrt(gruenblauI)
#t = matrix2latex(m, headerRow=hr, format='%.2f')
#print(t)

# Die erste violette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, violett1U, np.sqrt(violett1I))
a, b = correlated_values(params, covariance_matrix)
U_g[3] = -b/a
print('U_g für erste violette =', -b/a)
violett1linspace = np.linspace(-0.1, 1.35, 500)

plt.plot(violett1U, np.sqrt(violett1I), 'rx', label='Messwerte', mew=0.5)
plt.plot(violett1linspace, linearfit(violett1linspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.3, -1, 26])
plt.savefig('build/violett1.pdf')
plt.clf()

hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
m = np.zeros((np.size(violett1U), 3))
m[:,0] = violett1U
m[:,1] = violett1I
m[:,2] = np.sqrt(violett1I)
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

# Die zweite violette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, violett2U, np.sqrt(violett2I))
a, b = correlated_values(params, covariance_matrix)
U_g[4] = -b/a
print('U_g für zweite violette =', -b/a)
violett2linspace = np.linspace(-0.1, 1.5, 500)

plt.plot(violett2U, np.sqrt(violett2I), 'rx', label='Messwerte', mew=0.5)
plt.plot(violett2linspace, linearfit(violett2linspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.5, -1, 16.5])
plt.savefig('build/violett2.pdf')
plt.clf()

hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
m = np.zeros((np.size(violett2U), 3))
m[:,0] = violett2U
m[:,1] = violett2I
m[:,2] = np.sqrt(violett2I)
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

# Die ultraviolette Spektrallinie

params, covariance_matrix = optimize.curve_fit(linearfit, ultraviolettU, np.sqrt(ultraviolettI))
a, b = correlated_values(params, covariance_matrix)
U_g[5] = -b/a
print('U_g für ultraviolette =', -b/a)
ultraviolettlinspace = np.linspace(-0.1, 1.95, 500)

plt.plot(ultraviolettU, np.sqrt(ultraviolettI), 'rx', label='Messwerte', mew=0.5)
plt.plot(ultraviolettlinspace, linearfit(ultraviolettlinspace, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)

plt.xlabel(r'$U/$V')
plt.ylabel(r'$\sqrt{I}/\mathrm{\sqrt{pA}}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([-0.05, 1.8, -1, 19])
plt.savefig('build/ultraviolett.pdf')
plt.clf()

hr = ['$U/$V', '$I/$A', '$\sqrt{I}/\mathrm{\sqrt{pA}}$']
m = np.zeros((np.size(ultraviolettU), 3))
m[:,0] = ultraviolettU
m[:,1] = ultraviolettI
m[:,2] = np.sqrt(ultraviolettI)
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

# Dicke Ausgleichsrechnung

lambbda = np.array([578, 546, 492, 435, 406, 365.5])
lambbda *= 1e-9
c = 299792458
f = c/lambbda
flin = np.linspace(0,1,1000)

params, covariance_matrix = optimize.curve_fit(linearfit, f, unp.nominal_values(U_g))
a, b = correlated_values(params, covariance_matrix)
print('Unser Wert für h/e ist', a)
print('Unser Wert für A_k ist', b)

plt.plot(f*1e-15, unp.nominal_values(U_g), 'rx', label='Messwerte', mew=0.5)
plt.plot(flin, linearfit(flin*1e15, *params), 'r-', label='Ausgleichsfunktion', linewidth=0.5)
plt.xlabel(r'$f \cdot 10^{-15}/$Hz')
plt.ylabel(r'$U/$V')
plt.tight_layout()
plt.legend()
plt.grid()
plt.axis([0, 1, -1.7, 2.5])
plt.savefig('build/dickfett.pdf')
plt.clf()

# Grosse Untersuchung der gelben Spektrallinie

U, I = np.genfromtxt('data/spannung.txt', unpack=True)
plt.plot(U, I, 'rx', label='Messwerte', mew=0.5)
plt.xlabel(r'$U/$V')
plt.ylabel(r'$I/$pA')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/spannung.pdf')
plt.clf()

#hr = ['$U/$V', '$I/$A']
#m = np.zeros((np.size(U), 2))
#m[:, 0] = U
#m[:, 1] = I
#t = matrix2latex(m, headerRow=hr, format='%.2f')
#print(t)
