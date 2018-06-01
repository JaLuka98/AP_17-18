import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex

def expfit(x,N0, mu):
    return N0*np.exp(-mu*x)

def linfit(x,a,b):
    return a*x+b

#Nullmessung
tno=1000  #s
Nno=1028
Ano = Nno/tno # Nullaktivität

#Messung des Strahlers ohne Absorber
tnm=60 #s
Nnm=8808

# Zink

e = 1.295
eins = (1+e)/e**2
zwei = eins*(2*(1+e)/(1+2*e)-np.log(1+2*e)/e)
drei = np.log(1+2*e)/(2*e)
vier = -(1+3*e)/(1+2*e)**2
sigmacompton = 2*np.pi*(2.81794032*1e-15)**2 * (zwei+drei+vier)
print(30*6.022140857*1e23*sigmacompton/(9.157*1e-6))

d, t, N = np.genfromtxt('data\zink.txt', unpack=True)  # d in mm, t in s
Azink = N/t - Ano

# Az = N/t - An0
hr = ['$d/$mm', '$t/$s', '$N_\mathrm{Zn} \pm \sigma N_\mathrm{Zn}$', '$(A_\mathrm{Zn} \pm \sigma N_\mathrm{Zn})\cdot $s']
m = np.zeros((np.size(d), 4))
m[:, 0] = d
m[:, 1] = t
m[:, 2] = N
m[:, 3] = Azink
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)

sigmaAno = np.sqrt(Nno)/tno
sigmaA = np.sqrt(N)/t
print(sigmaA-sigmaAno)

params, covariance_matrix = optimize.curve_fit(expfit, d, Azink)

N0, mu = correlated_values(params, covariance_matrix)

print('N0 =', N0)
print('mu in 1/mm =', mu*1e3)

dlin = np.linspace(0, 22)
plt.errorbar(d, Azink, yerr=sigmaA-sigmaAno, fmt = 'rx', mew = 0.5, ecolor='blue', label='Messwerte')
plt.plot(dlin, expfit(dlin, *params), 'r-', label='Ausgleichsrechnung')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$d/$mm')
plt.ylabel(r'$A_\mathrm{zn} \,\cdot\, $s')
plt.tight_layout()
plt.axis((0,22,50,145))
plt.legend()
plt.grid()
plt.savefig('build/zink.pdf')
plt.clf()

# Eisen

print(26*6.022140857*1e23*sigmacompton/(7.0923*1e-6))
d, t, N = np.genfromtxt('data\eisen.txt', unpack=True)  # d in mm, t in s
Aeisen = N/t - Ano

# Az = N/t - An0
hr = ['$d/$mm', '$t/$s', '$N_\mathrm{Fe} \pm \sigma N_\mathrm{Fe}$', '$(A_\mathrm{Fe} \pm \sigma N_\mathrm{Fe})\cdot $s']
m = np.zeros((np.size(d), 4))
m[:, 0] = d
m[:, 1] = t
m[:, 2] = N
m[:, 3] = Aeisen
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)

sigmaAno = np.sqrt(Nno)/tno
sigmaA = np.sqrt(N)/t
print(sigmaA-sigmaAno)

params, covariance_matrix = optimize.curve_fit(expfit, d, Aeisen)

N0, mu = correlated_values(params, covariance_matrix)

print('N0 =', N0)
print('mu in 1/mm =', mu*1e3)

dlin = np.linspace(0, 70)
plt.errorbar(d, Aeisen, yerr=sigmaA-sigmaAno, fmt = 'rx', mew = 0.5, ecolor='blue', label='Messwerte')
plt.plot(dlin, expfit(dlin, *params), 'r-', label='Ausgleichsrechnung')
plt.yscale('log', nonposy='clip')
plt.xlabel(r'$d/$mm')
plt.ylabel(r'$A_\mathrm{Fe} \,\cdot\, $s')
plt.tight_layout()
plt.axis((0,70,0,200))
plt.legend()
plt.grid()
plt.savefig('build/eisen.pdf')
plt.clf()

#Betastrahlung

d, derror, t, N, A, sigmaA = np.genfromtxt('data/beta.txt', unpack=True)
sigmaA = np.sqrt(N)/t
R = d*1e-6*2700 # Massenbelegung
Rerror = derror*1e-6*2700
print(sigmaA)

hr = ['$d/$mm', '$t/$s', '$N_\beta \pm \sigma N_beta$', '$(A_\beta \pm \sigma N_\beta)\,\cdot\, $s']
m = np.zeros((np.size(d), 4))
m[:, 0] = d
m[:, 1] = t
m[:, 2] = N
m[:, 3] = A
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)

params1, covariance_matrix1 = optimize.curve_fit(linfit, R[0:3], np.log(A[0:3]))
a1, b1 = correlated_values(params1, covariance_matrix1)
print('a1 =', a1)
print('b1 =', b1)

params2, covariance_matrix2 = optimize.curve_fit(linfit, R[6:], np.log(A[6:]))
a2, b2 = correlated_values(params2, covariance_matrix2)
print('a2 =', a2)
print('b2 =', b2)

Rmax = (b2-b1)/(a1-a2)
Rmax *= 1e-4*1e3 # in g/cm^2
Emax = 1.92 * unp.sqrt(Rmax**2+0.22*Rmax)
print(Rmax)
print(Emax)

R1lin = np.linspace(0.2, 0.9, 1000)
R2lin = np.linspace(0.2, 1.3, 1000)
plt.errorbar(R[0:3], np.log(A[0:3]), xerr=Rerror[0:3],
             yerr=[np.log(A[0:3]+sigmaA[0:3])-np.log(A[0:3]), np.log(A[0:3])-np.log(A[0:3]-sigmaA[0:3])],
             fmt = 'rx', mew = 0.5, ecolor='blue', label='Messwerte 1')
plt.errorbar(R[3:5], np.log(A[3:5]), xerr=Rerror[3:5],
             yerr=[np.log(A[3:5]+sigmaA[3:5])-np.log(A[3:5]), np.log(A[3:5])-np.log(A[3:5]-sigmaA[3:5])],
             fmt = 'x', color = 'black', mew = 0.5, ecolor='black', label='ausgenommene Messwerte')
plt.errorbar(R[6:], np.log(A[6:]), xerr=Rerror[6:],
             yerr=[np.log(A[6:]+sigmaA[6:])-np.log(A[6:]), np.log(A[6:])-np.log(A[6:]-sigmaA[6:])],
             fmt = 'x', color = 'orchid', mew = 0.5, ecolor='indigo', label='Messwerte 2')
plt.plot(R1lin, linfit(R1lin, *params1), 'r-', label='Ausgleichsrechnung 1')
plt.plot(R2lin, linfit(R2lin, *params2), '-', color = 'orchid', label='Ausgleichsrechnung 2')
#plt.yscale('log', nonposy='clip')
plt.xlabel(r'$R/\frac{\mathrm{kg}}{\mathrm{m^2}}$')
plt.ylabel(r'$\ln{(A_\beta \cdot \mathrm{s})}$')
plt.tight_layout()
plt.axis((0.2,1.3,-1,4))
plt.legend()
plt.grid()
plt.savefig('build/beta.pdf')
plt.clf()
print(R[0:2])
