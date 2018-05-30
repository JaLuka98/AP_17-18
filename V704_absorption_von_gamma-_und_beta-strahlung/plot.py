import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from matrix2latex import matrix2latex

#Nullmessung
tno=1000  #s
Nno=1028
Ano = Nno/tno # Nullaktivität

#Messung des Strahlers ohne Absorber
tnm=60 #s
Nnm=8808

# Zink

d, t, N = np.genfromtxt('data\zink.txt', unpack=True)  # d in mm, t in s
Azink = N/t - Ano

# Az = N/t - An0
hr = ['$d/$mm', '$t/$s', '$N_\mathrm{z} \pm \sigma N_\mathrm{z}$', '$(A_\mathrm{z}) \pm \sigma N_\mathrm{z})\cdot $s']
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
