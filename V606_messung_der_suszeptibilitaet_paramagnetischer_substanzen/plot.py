import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex

F=86.6*1e-6 #m^2
U_sp=1 #laut versuchsanleitung soll das 1 sein... haben wir aber nicht gemessen
mu_B=1.602*1e-19*6.626*1e-34/(4*np.pi*9.11*1e-31)
T=293.15 #Kelvin
k_B=1.3805*1e-23 #J/K
N_A=6.022*1e23
R3=998 #Ohm

f, U = np.genfromtxt('data/glocke.txt', unpack=True) #f/kHz, U/V


plt.plot(f, U, 'rx', mew=0.5)
plt.xlabel(r'$f/$kHz')
plt.ylabel(r'$U_a/$V')
plt.tight_layout()
plt.legend()
plt.grid()


plt.axes([0.21, 0.39, 0.4, 0.45])
plt.plot(f,U, 'rx', mew=0.5)
plt.xlim(34, 36)
plt.ylim(0, 1.95)
#plt.xlabel(r'$f/$Hz')
#plt.ylabel(r'$U_a/$V')
plt.tight_layout()
plt.grid()
plt.savefig('build/glocke.pdf')
plt.clf()

hr = ['$f/$kHz','$U_a/$V']
m = np.zeros((45, 2))
m[:,0] = f
m[:,1] = U
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)


#DY2O3
print('Auswertung zu DY2O6:')

m1=15.1*1e-3 #kg
L1=0.135     #ungefähr gleich der Länge der Spule
rho1=7800 #kg/m^3

Q1=m1/(L1*rho1)
print('Q = ', Q1)

U_o1=16*1e-3
U_o2=16*1e-3
U_o3=16*1e-3
U_m1=34*1e-3
U_m2=33*1e-3
U_m3=33.5*1e-3
R_o1=710*5e-3
R_o2=707*5e-3
R_o3=724*5e-3
R_m1=328*5e-3
R_m2=340*5e-3
R_m3=351*5e-3

U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m

#deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
U_mmittel=ufloat(np.mean(U_m), np.std(U_m))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))
#R_mmittel = ufloat(np.mean(R_m), np.std(R_m))

X_U=4*F*U_mmittel/(Q1*U_sp)*1e-2  #GAINS!!!
X_R=2*deltaRmittel/R3*F/Q1 #Hier wusste ich nicht genau was hier was sein soll, aber Lucas meinte das muss so
print('Die Suszeptibilität aus der Spannung beträgt:', X_U)
print('Die Suszeptibilität aus dem Widerstand beträgt:', X_R)

hr = ['$U_o/$mV','$U_m/$mV','$R_o4/\Omega$','$R_m/\Omega$', '$\Delta R/\Omega$']
m = np.zeros((3, 5))
m[:,0] = U_o*1e3
m[:,1] = U_m*1e3
m[:,2] = R_o
m[:,3] = R_m
m[:,4] = deltaR
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)


#Berechnung des Theoriewerts
J=7.5
L=5
S=2.5
g_j=(4/3)
M1=372.998*1e-3

N1=2*N_A*rho1/M1 #Vielleicht auch noch mit Faktor zwei....?!

X_T1=4*np.pi*1e-7*(mu_B)**2*(g_j)**2*N1*J*(J+1)/(3*k_B*T)
print('Der Theoriewert ist:', X_T1)




#C6O12Pr2
print('Auswertung zu C6O12Pr2:')
m2=7.87*1e-3
L2=0.135     #ungefähr gleich der Länge der Spule
#rho2=? #Da müssen wir wohl google fragen... Google sagt da nichts zu...

#Q2=m2/(L2*rho2)

U_o1=14.5*1e-3
U_o2=16*1e-3
U_o3=15.5*1e-3
U_m1=16*1e-3
U_m2=16*1e-3
U_m3=16*1e-3
R_o1=639*5e-3
R_o2=627*5e-3
R_o3=633*5e-3
R_m1=600*5e-3
R_m2=614*5e-3
R_m3=560*5e-3
U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m

#deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
U_mmittel=ufloat(np.mean(U_m), np.std(U_m))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))
#R_mmittel = ufloat(np.mean(R_m), np.std(R_m))
hr = ['$U_o/$mV','$U_m/$mV','$R_o4/\Omega$','$R_m/\Omega$', '$\Delta R/\Omega$']
m = np.zeros((3, 5))
m[:,0] = U_o*1e3
m[:,1] = U_m*1e3
m[:,2] = R_o
m[:,3] = R_m
m[:,4] = deltaR
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)


#Berechnung des Theoriewerts
J=4
L=5
S=1
g_j=4/5
M2=545.8723*1e-3

#N2=2*N_A*rho2/M2 #Vielleicht auch noch mit Faktor zwei....?!

#X_T2=4*np.pi*1e-7*(mu_B)**2*(g_j)**2*N2*J*(J+1)/(3*k_B*T)
#print('Der Theoriewert ist:', X_T2)


#Gd2O3
print('Auswertung zu Gd203:')
m3=14.08*1e-3
L3=0.135     #ungefähr gleich der Länge der Spule
rho3=7400 #kg/m^3

Q3=m3/(L3*rho3)

U_o1=16*1e-3
U_o2=17.5*1e-3
U_o3=17*1e-3
U_m1=21.5*1e-3
U_m2=21.5*1e-3
U_m3=20.5*1e-3
R_o1=293*5e-3
R_o2=294*5e-3
R_o3=297*5e-3
R_m1=137*5e-3
R_m2=146*5e-3
R_m3=147*5e-3


U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m

#deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
U_mmittel=ufloat(np.mean(U_m), np.std(U_m))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))
#R_mmittel = ufloat(np.mean(R_m), np.std(R_m))

X_U=4*F*U_mmittel/(Q3*U_sp)*1e-2  #GAINS!!!
X_R=2*deltaRmittel/R3*F/Q3 #Hier wusste ich nicht genau was hier was sein soll, aber Lucas meinte das muss so
print('Die Suszeptibilität aus der Spannung beträgt:', X_U)
print('Die Suszeptibilität aus dem Widerstand beträgt:', X_R)

hr = ['$U_o/$mV','$U_m/$mV','$R_o4/\Omega$','$R_m/\Omega$', '$\Delta R/\Omega$']
m = np.zeros((3, 5))
m[:,0] = U_o*1e3
m[:,1] = U_m*1e3
m[:,2] = R_o
m[:,3] = R_m
m[:,4] = deltaR
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)


#Berechnung des Theoriewerts
J=3.5
L=0
S=3.5
g_j=2
M3=362.4982*1e-3

N3=2*N_A*rho3/M3 #Vielleicht auch noch mit Faktor zwei....?!

X_T3=4*np.pi*1e-7*(mu_B)**2*(g_j)**2*N3*J*(J+1)/(3*k_B*T)
print('Der Theoriewert ist:', X_T3)



#Nd2O3
print('Auswertung zu Nd203:')
m4=9*1e-3
L4=0.135     #ungefähr gleich der Länge der Spule
rho4=7240 #kg/m^3

Q4=m4/(L4*rho4)

U_o1=16.5*1e-3
U_o2=18.25*1e-3
U_o3=17.75*1e-3
U_m1=18*1e-3
U_m2=18.25*1e-3
U_m3=18*1e-3
R_o1=320*5e-3
R_o2=323*5e-3
R_o3=323*5e-3
R_m1=297*5e-3
R_m2=286*5e-3
R_m3=296*5e-3

U_o=np.array([U_o1, U_o2, U_o3])
U_m=np.array([U_m1, U_m2, U_m3])
R_o=np.array([R_o1, R_o2, R_o3])
R_m=np.array([R_m1, R_m2, R_m3])

deltaU=U_m-U_o
deltaR=R_o-R_m

#deltaUmittel = ufloat(np.mean(deltaU), np.std(deltaU))
U_mmittel=ufloat(np.mean(U_m), np.std(U_m))
deltaRmittel = ufloat(np.mean(deltaR), np.std(deltaR))
#R_mmittel = ufloat(np.mean(R_m), np.std(R_m))

X_U=4*F*U_mmittel/(Q4*U_sp)*1e-2  #GAINS!!!
X_R=2*deltaRmittel/R3*F/Q4 #Hier wusste ich nicht genau was hier was sein soll, aber Lucas meinte das muss so
print('Die Suszeptibilität aus der Spannung beträgt:', X_U)
print('Die Suszeptibilität aus dem Widerstand beträgt:', X_R)

hr = ['$U_o/$mV','$U_m/$mV','$R_o4/\Omega$','$R_m/\Omega$', '$\Delta R/\Omega$']
m = np.zeros((3, 5))
m[:,0] = U_o*1e3
m[:,1] = U_m*1e3
m[:,2] = R_o
m[:,3] = R_m
m[:,4] = deltaR
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

#Berechnung des Theoriewerts
J=4.5
L=6
S=1.5
g_j=8/11
M4=336.4822*1e-3

N4=2*N_A*rho4/M4 #Vielleicht auch noch mit Faktor zwei....?!

X_T4=4*np.pi*1e-7*(mu_B)**2*(g_j)**2*N4*J*(J+1)/(3*k_B*T)
print('Der Theoriewert ist:', X_T4)



m_i=np.array([m1, m3, m4])
rho=np.array([rho1, rho3, rho4])
Q=np.array([Q1, Q3, Q4])


hr = ['$m/$g','$rho/frac{\symup{g}}{\symup{cm}^3}$','$Q/\symup{mm}^2$']
m = np.zeros((3, 3))
m[:,0] = m_i*1e3
m[:,1] = rho*1e-3
m[:,2] = Q*1e6
t = matrix2latex(m, headerRow=hr, format='%.2f')
print(t)
