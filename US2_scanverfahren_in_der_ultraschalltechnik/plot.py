import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex


#Schieblehre:
r1=r2=1.5*1e-3 #m
d12=1.5*1e-3 #m  Abstand der Beiden Störstellen
h=8.03*1e-2    #m Höhe des Acrylblocks

#A-Scan:
h_A=80.7*1e-3 #m mit Ultraschall gemessene Höhe des Acrylblocks

#B-Scan:
d12_b1=1.7*1e-3
#d12_b2 nicht bestimmbar

#TM-Scan
d_ohneluft=44.1*1e-3 #m
d_mitluft=32.5*1e-3 #m

r_z=(46.4/2)*1e-3 #m

c_W=1485
c_A=2730

#Jetzt zur Auswertung

#A_scan
nummer, s1_a, s2_a = np.genfromtxt('data/ascan.txt', unpack=True) #Daten A-Scan
s1_a-=0.4 #Laufzeitkorrektur
s2_a-=0.4
s1_a*=1e-3  #Umrechnung in m
s2_a*=1e-3

nummer, s1_s, s2_s, d_s = np.genfromtxt('data/schieblehre.txt', unpack=True) #Daten Schieblehre
s1_s*=1e-3  #Umrechnung in m
s2_s*=1e-3
d_s*=1e-3

d_a=h-s1_a-s2_a

deltad=np.absolute(d_s-d_a)

hr = ['Störstelle', '$s_{\symup{1,A}}/$mm', '$s_{\symup{2,A}}/$mm', '$d_{\symup{A}}/$mm',
'$s_{\symup{1,S}}/$mm','$s_{\symup{2,S}}/$mm','$d_{\symup{S}}/$mm','$\Delta d/mm$']
m = np.zeros((9, 8))
m[:, 0] = nummer
m[:, 1] = s1_a*1e3
m[:, 2] = s2_a*1e3
m[:, 3] = d_a*1e3
m[:, 4] = s1_s*1e3
m[:, 5] = s2_s*1e3
m[:, 6] = d_s*1e3
m[:, 7] = deltad*1e3
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)


#B-Scan

nummer, s1_b, s2_b = np.genfromtxt('data/bscan.txt', unpack=True) #Daten B-Scan
s1_b-=0.5 #Laufzeitkorrektur
s2_b-=0.5
s1_b*=1e-3  #Umrechnung in m
s2_b*=1e-3

d_b=h-s1_b-s2_b

deltad=np.absolute(d_s-d_b)

hr = ['Störstelle', '$s_{\symup{1,A}}/$mm', '$s_{\symup{2,A}}/$mm', '$d_{\symup{A}}/$mm','$\Delta d/mm$']
m = np.zeros((9, 5))
m[:, 0] = nummer
m[:, 1] = s1_b*1e3
m[:, 2] = s2_b*1e3
m[:, 3] = d_b*1e3
m[:, 4] = deltad*1e3
table = matrix2latex(m, headerRow=hr, format='%.2f')
print(table)


#Herzfrequenz
s_s = np.genfromtxt('data/herz.txt', unpack=True) #Daten Herzfrequenz
s_s*=(c_W/c_A) #Korrektur falscher Wert
s_s*=1e-3   #Umrechnung in Meter

h=d_ohneluft-s_s

V_s=(h*np.pi/6)*(3*r_z**2+h**2)

hr = ['$s/$mm','$h/$mm','$V/cm^3$']
m = np.zeros((17, 3))
m[:, 0] = s_s*1e3
m[:, 1] = h*1e3
m[:, 2] = V_s*1e6
table = matrix2latex(m, headerRow=hr, format='%.1f')
print(table)
