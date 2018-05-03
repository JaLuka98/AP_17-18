import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex

def linearfit(x, a, b):
    return a*x+b

#E-Feld Auswertung

U_b1=180    #Beschleunigungsspannungen in V
U_b2=230
U_b3=270
U_b4=300
U_b5=350

v_1=39.94   #Frequenzen in Hz
v_2=79.87
v_3=159.70
v_4=239.50

A=7.5       #Amplitude in mm


print('Die Berechnungen fürs E-Feld:')

U_d1, U_d2, U_d3, U_d4 ,U_d5, D = np.genfromtxt('data/efeld.txt', unpack=True) #U in V und D in mm
D*=1e-3 #D in m

params, covariance_matrix = optimize.curve_fit(linearfit, U_d1, D)
errors = np.sqrt(np.diag(covariance_matrix))
print('D/U_d1 =', params[0]*1e3, '+-', errors[0]*1e3)
E1= ufloat(params[0], errors[0])
b1 = params[0]  #brauchen wir nachher noch...

plt.plot(U_d1, D, 'rx', mew=0.5, label='Messwerte')
plt.plot(U_d1, linearfit(U_d1, *params), 'r-',linewidth=0.5, label='Ausgleichsfunktion')


params, covariance_matrix = optimize.curve_fit(linearfit, U_d2, D)
errors = np.sqrt(np.diag(covariance_matrix))
print('D/U_d2 =', params[0]*1e3, '+-', errors[0]*1e3)
E2 = ufloat(params[0], errors[0])
b2 = params[0] #brauchen wir nachher noch...

plt.plot(U_d2, D, 'bx',mew=0.5, label='Messwerte')
plt.plot(U_d2, linearfit(U_d2, *params), 'b-', linewidth=0.5, label='Ausgleichsfunktion')


params, covariance_matrix = optimize.curve_fit(linearfit, U_d3, D)
errors = np.sqrt(np.diag(covariance_matrix))
print('D/U_d3 =', params[0]*1e3, '+-', errors[0]*1e3)
E3 = ufloat(params[0], errors[0])
b3 = params[0]

plt.plot(U_d3, D, 'gx',mew=0.5, label='Messwerte')
plt.plot(U_d3, linearfit(U_d3, *params), 'g-', linewidth=0.5, label='Ausgleichsfunktion')

plt.xlabel(r'$U_d/$V')
plt.ylabel(r'$D/$m')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/ablenkspannung.pdf')
plt.clf()

#Alle in einem Plot wäre wahrscheinlich ein bisschen viel....

params, covariance_matrix = optimize.curve_fit(linearfit, U_d4, D)
errors = np.sqrt(np.diag(covariance_matrix))
print('D/U_d4 =', params[0]*1e3, '+-', errors[0]*1e3)
E4 = ufloat(params[0], errors[0])
b4 = params[0]

plt.plot(U_d4, D, 'yx', mew=0.5, label='Messwerte')
plt.plot(U_d4, linearfit(U_d4, *params), 'y-', linewidth=0.5, label='Ausgleichsfunktion')


params, covariance_matrix = optimize.curve_fit(linearfit, U_d5, D)
errors = np.sqrt(np.diag(covariance_matrix))
print('D/U_d5 =', params[0]*1e3, '+-', errors[0]*1e3)
E5 = ufloat(params[0], errors[0])
b5 = params[0]

plt.plot(U_d5, D, 'mx', mew=0.5, label='Messwerte')
plt.plot(U_d5, linearfit(U_d5, *params), 'm-', linewidth=0.5, label='Ausgleichsfunktion')

plt.xlabel(r'$U_d/$V')
plt.ylabel(r'$D/$m')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/ablenkspannung2.pdf')
plt.clf()

hr = ['$U_d1/$V','$U_d2/$V','$U_d3/$V','$U_d4/$V','$U_d5/$V', '$D/$m']
m = np.zeros((9, 6))
m[:,0] = U_d1
m[:,1] = U_d2
m[:,2] = U_d3
m[:,3] = U_d4
m[:,4] = U_d5
m[:,5] = D*1e3
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

#Im folgenden Abschnitt nehme ich die Fehler nicht mit... geht das vielleicht irgendwie schöner?

E=np.array([b1, b2, b3, b4, b5])
U_b=np.array([U_b1, U_b2, U_b3, U_b4, U_b5])

params, covariance_matrix = optimize.curve_fit(linearfit, 1/U_b, E)
errors = np.sqrt(np.diag(covariance_matrix))
print('a =', params[0], '+-', errors[0])
a = ufloat(params[0], errors[0])


plt.plot((1/U_b)*1e3, E*1e3, 'rx', mew=0.5, label='Messwerte')
plt.plot((1/U_b)*1e3, linearfit(1/U_b, *params)*1e3, 'b-', linewidth=0.5, label='Ausgleichsfunktion')

plt.xlabel(r'$\frac{1}{U_b}/\frac{1}{V}\cdot 10^{-3}$')
plt.ylabel(r'$\frac{D}{U_d}/\frac{m}{V}\cdot 10^{-3}$')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/empfindlichkeiten.pdf')
plt.clf()

#Diese komische Größe berechnen

#Da es um y-Ablenkung geht, nehme ich auch mal die Werte dafür aus der Zeichnung
p=1.9 #cm
L=14.3 #cm
d=0.38  #cm

p*=1e-2 #alles in meter
L*=1e-2
d*=1e-2

a_zeichnung=p*L/(2*d)

print('Die Größe pL/(2d) ist a_zeichnung=', a_zeichnung)


#B-Feld Auswertung
print('Jetzt die Berechnungen fürs B-Feld:')
U_b1=250    #Beschleunigungsspannungen in V
U_b2=400
L=0.175     #Strecke, auf der die Elektronen abgelenkt werden in m
N=20        #Anzahl der Windungen

R=0.282 #Werte von Kevin :D

r=D/(L**2+D**2)

I_1, I_2, D= np.genfromtxt('data/bfeld.txt', unpack=True) #I/A, D/mm
B_1=4*np.pi*1e-7*8/(np.sqrt(125))*N*I_1/R
B_2=4*np.pi*1e-7*8/(np.sqrt(125))*N*I_2/R

params, covariance_matrix = optimize.curve_fit(linearfit, B_1, r)
errors = np.sqrt(np.diag(covariance_matrix))
print('a_1 =', params[0], '+-', errors[0])
a1 = ufloat(params[0], errors[0])

plt.plot(B_1*1e6, r, 'bx', mew=0.5, label='Messwerte')
plt.plot(B_1*1e6, linearfit(B_1, *params), 'b-', linewidth=0.5, label='Ausgleichsfunktion')


params, covariance_matrix = optimize.curve_fit(linearfit, B_2, r)
errors = np.sqrt(np.diag(covariance_matrix))
print('a_2 =', params[0], '+-', errors[0])
a2 = ufloat(params[0], errors[0])

plt.plot(B_2*1e6, r, 'rx', mew=0.5, label='Messwerte')
plt.plot(B_2*1e6, linearfit(B_2, *params), 'r-', linewidth=0.5, label='Ausgleichsfunktion')

plt.xlabel(r'$B$/µT')
plt.ylabel(r'$r$/m')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/bfeld.pdf')
plt.clf()

print('e/m_1 ergibt:', a1**2*8*U_b1)
print('e/m_1 ergibt:', a2**2*8*U_b2)

hr = ['$I_1/$A','$B_1/$µT','$I_2/$A','$B_2/$µT', '$D/$mm','$r/$m' ]
m = np.zeros((9, 6))
m[:,0] = I_1
m[:,1] = B_1*1e6
m[:,2] = I_2
m[:,3] = B_2*1e6
m[:,4] = D
m[:,5] = r
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

#Das Erdmagnetfeld

I_erde=50*1e-3 #Stromstärke in A
alpha=72          #Winkel in Grad
phi=alpha*2*np.pi/360

B_spulen=4*np.pi*1e-7*8/(np.sqrt(125))*N*I_erde/R
print('B_gegen=', B_spulen)
B_erde=B_spulen/np.cos(phi)
print('Das Erdnamagnetfeld: B_erde=', B_erde)
