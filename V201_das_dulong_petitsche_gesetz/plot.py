import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat

T_x=273.15+21.6    #K
T_y=273.15+81.0    #K
T_m=273.15+45.9    #K

c_w=4.18        #j/g*K

m_y=493.69-233.45
m_x=834.46-233.45-m_y
M_ges=m_x+m_y

print('T_x = ', T_x)
print('T_y = ', T_y)
print('T_m = ', T_m)
print('m_x = ', m_x)
print('m_y = ', m_y)
print('m_ges = ', M_ges)
print('c_g*m_g = ', (c_w*m_y*(T_y - T_m)-c_w*m_x*(T_m - T_x))/(T_m - T_x))

cm_g=(c_w*m_y*(T_y - T_m)-c_w*m_x*(T_m - T_x))/(T_m - T_x)

m_wblei=826.85-233.45
m_walu=839.47-238.70
m_wkupfer=858.27-238.70

print('m_wblei = ', m_wblei)
print('m_walu= ',m_walu)
print('m_wkupfer = ', m_wkupfer)

m_blei=676.07-5.25-140.07
m_alu=300.43-144.2
m_kupfer=377.66-140.05

print('m_blei = ', m_blei)
print('m_alu= ',m_alu)
print('m_kupfer = ', m_kupfer)

T_w, T_k, T_m = np.genfromtxt('data/Blei.txt', unpack=True)

T_w+=273.15 #K
T_k+=273.15 #K
T_m+=273.15 #K
c_k=((c_w*m_wblei+cm_g)*(T_m-T_w))/(m_blei*(T_k-T_m))
c_k_m = ufloat(np.sum(c_k)/len(c_k), np.std(c_k))
T_m_m = ufloat(np.sum(T_m)/len(T_m), np.std(T_m))
mol=c_k_m*207.2-((9*((29)**2)*207.2*T_m_m)*1e-9/(11.35*42))

print('BLEI')
print('T_w = ', T_w)
print('T_k = ', T_k)
print('T_m = ', T_m)
print('c_k = ', c_k)
print('Mittelwert von c_k: ', c_k_m)
print('Mittelwert von T_m:', T_m_m)
print('Molwärme: ', mol)


T_w, T_k, T_m = np.genfromtxt('data/Aluminium.txt', unpack=True)

T_w+=273.15 #K
T_k+=273.15 #K
T_m+=273.15 #K
c_k=((c_w*m_walu+cm_g)*(T_m-T_w))/(m_alu*(T_k-T_m))
c_k_m = ufloat(np.sum(c_k)/len(c_k), np.std(c_k))
T_m_m = ufloat(np.sum(T_m)/len(T_m), np.std(T_m))
mol=c_k_m*27-((9*((23.5)**2)*27*T_m_m)*1e-9/(2.7*75))

print('ALUMINIUM')
print('T_w = ', T_w)
print('T_k = ', T_k)
print('T_m = ', T_m)
print('c_k = ', c_k)
print('Mittelwert von c_k: ', c_k_m)
print('Mittelwert von T_m:', T_m_m)
print('Molwärme: ', mol)


T_w, T_k, T_m = np.genfromtxt('data/Kupfer.txt', unpack=True)

T_w+=273.15 #K
T_k+=273.15 #K
T_m+=273.15 #K
c_k=((c_w*m_wkupfer+cm_g)*(T_m-T_w))/(m_kupfer*(T_k-T_m))
c_k_m = ufloat(np.sum(c_k)/len(c_k), np.std(c_k))
T_m_m = ufloat(np.sum(T_m)/len(T_m), np.std(T_m))
mol=c_k_m*63.5-((9*((16.8)**2)*63.5*T_m_m)*1e-9/(8.96*136))

print('KUPFER')
print('T_w = ', T_w)
print('T_k = ', T_k)
print('T_m = ', T_m)
print('c_k = ', c_k)
print('Mittelwert von c_k: ', c_k_m)
print('Mittelwert von T_m:', T_m_m)
print('Molwärme: ', mol)
