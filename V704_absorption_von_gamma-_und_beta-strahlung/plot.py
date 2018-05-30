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

# theoretische Berechnung

epsilon = 1.295

sigmacompton = 2*np.pi*(2.81794032*1e-15)*2 * ((1+epsilon)/epsilon**2 *
               ((2*(1+epsilon))/(1+2*epsilon) - np.log(2+2*epsilon)/epsilon) +
               np.log(1+2*epsilon)/(2*epsilon)-(1+3*epsilon)/(1+2*epsilon)**2)
print(sigmacompton)
print(30*6.022140857*1e23*sigmacompton/(9.16*1e-6))

d, t, N = np.genfromtxt('data\zink.txt', unpack=True)  # d in mm, t in s
Azink = N/t - Ano

# Az = N/t - An0
hr = ['$d/$mm', '$t/$s', '$N_\mathrm{z} \pm \sigma N_\mathrm{z}$', '$(A_\mathrm{z} \pm \sigma N_\mathrm{z})\cdot $s']
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
print('mu=', mu)

plt.semilogy(d, Azink, 'rx', mew=0.5)
plt.xlabel(r'$d/$mm')
plt.ylabel(r'$A_\mathrm{z} \,\cdot\, $s')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/zink.pdf')
plt.clf()
