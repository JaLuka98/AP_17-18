import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex

t=120 #s
p_0=1.013 #bar
d_1=2.2 #cm
d_2=1.8 #cm

d1*=1e-2    #in meter
d2*=1e-2    #in meter


#Erste Messung
print('Erste Messung:')
p, channel, N_ges  = np.genfromtxt('data/druck.txt', unpack=True)
p*=1e-3 #p in bar

x=d1*p/p_0
Z=N_ges/t

hr = ['p/'mbar,'channel','$N_{ges}$', '$x$/cm']
m = np.zeros((21, 4))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)




#Zweite Messung
print('Zweite Messung:')
p, channel, N-ges  = np.genfromtxt('data/druck2.txt', unpack=True)
p*=1e-3

x=d1*p/p_0
Z=N_ges/t

hr = ['p/'mbar,'channel','$N_{ges}$', '$x$/cm']
m = np.zeros((21, 3))
m[:,0] = p*1e3
m[:,1] = channel
m[:,2] = N_ges
m[:,3] = x*1e2
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)
