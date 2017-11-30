import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat



n1=300 #lange spule
I1=1.4 #Strom in A für lange spule
l1=0.16 #länge in m lange spule

n2=100 #kurze spule
I2=1.2 #strom kurze Spule
l2=0.055 #länge in m kurze Spule
D2=0.041 #durchmesser in m

r, B = np.genfromtxt('data/lange_spule.txt', unpack=True) #r/cm B/mT
r *= 1e-2 #r/m
B *= 1e-3 #B/T

x=np.linspace(1.5, 17.5)

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x, (4*np.pi*1e-7*n1*I1*1e3/l1)+0*x, 'b--', label = 'Theoriewert in der Spule')
plt.plot(x, (4*np.pi*1e-7*n1*I1*1e3/l1)+0*x-0.63, 'g--', label = 'Theoriewert mit angepasstem Nullniveau')
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/lange_spule.pdf')

plt.clf()

print('Erwartungswert für die lange Spule:', (4*np.pi*1e-7*n1*I1*1e3/l1), 'mT')

r, B = np.genfromtxt('data/kurze_spule.txt', unpack=True) #r/cm B/mT
r *= 1e-2 #r/m
B *= 1e-3 #B/T

x=np.linspace(1.5, 7)

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x, (4*np.pi*1e-7*n2*I2*1e3/(np.sqrt(l2*l2+D2*D2)))+0*x - 0.57, 'b--', label = 'Theoriewert in der Spule')
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/kurze_spule.pdf')

plt.clf()

print('Erwartungswert für die kurze Spule:', 4*np.pi*1e-7*n2*I2*1e3/(np.sqrt(l2*l2+D2*D2)), 'mT')

I_H1=3 #strom durch spulenpaar
R=0.0625 #spulenradius
n=100 #100 Windungen je spule

r, B = np.genfromtxt('data/spulenpaar_3.txt', unpack=True) #r/cm B/mT
r *= 1e-2 #r/m
B *= 1e-3 #B/T

x=np.linspace(-0.03125, 0.235)

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x*1e2, 1e3*((4*np.pi*1e-7*n*I_H1)/(2*R))*(1/((((x/R)**2)+(x/R)+(5/4))**(3/2))
+1/((((x/R)**2)-(x/R)+(5/4))**(3/2)))+0*x, 'b--', label = 'Theoriewert')
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
plt.legend()
plt.grid()

plt.axes([0.2, 0.3, 0.31, 0.31])

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x*1e2, 1e3*((4*np.pi*1e-7*n*I_H1)/(2*R))*(1/((((x/R)**2)+(x/R)+(5/4))**(3/2))
+1/((((x/R)**2)-(x/R)+(5/4))**(3/2)))+0*x, 'b--', label = 'Theoriewert')
plt.xlim(-1, 1)
plt.ylim(3,4.5)
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.savefig('build/spulenpaar_3.pdf')

plt.clf()

print('Erwartungswert für Helmholzspulenpaar mit 3A:', (1e3*(4*np.pi*1e-7*n*I_H1)/(2*R))*(1/((((0/R)**2)+(0/R)+(5/4))**(3/2))
+1/((((0/R)**2)-(0/R)+(5/4))**(3/2))), 'mT')

I_H2=5

r, B = np.genfromtxt('data/spulenpaar_5.txt', unpack=True) #r/cm B/mT
r *= 1e-2 #r/m
B *= 1e-3 #B/T

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x*1e2, 1e3*((4*np.pi*1e-7*n*I_H2)/(2*R))*(1/((((x/R)**2)+(x/R)+(5/4))**(3/2))
+1/((((x/R)**2)-(x/R)+(5/4))**(3/2)))+0*x, 'b--', label = 'Theoriewert')
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
plt.legend()
plt.grid()

plt.axes([0.2, 0.3, 0.31, 0.31])

plt.plot(r*1e2, B*1e3, 'rx', label='Messwerte')
plt.plot(x*1e2, 1e3*((4*np.pi*1e-7*n*I_H2)/(2*R))*(1/((((x/R)**2)+(x/R)+(5/4))**(3/2))
+1/((((x/R)**2)-(x/R)+(5/4))**(3/2)))+0*x, 'b--', label = 'Theoriewert')
plt.xlim(-1, 1)
plt.ylim(6,7.5)
plt.xlabel(r'$r/$cm')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
#plt.legend()
plt.grid()
plt.savefig('build/spulenpaar_5.pdf')

plt.clf()

print('Erwartungswert für Helmholzspulenpaar mit 5A:', (1e3*(4*np.pi*1e-7*n*I_H2)/(2*R))*(1/((((0/R)**2)+(0/R)+(5/4))**(3/2))
+1/((((0/R)**2)-(0/R)+(5/4))**(3/2))), 'mT')


I, B = np.genfromtxt('data/hysterese.txt', unpack=True) #I/a, B/mT
B*=1e-3 #B/T

plt.plot(I, B*1e3, 'rx', label='Messwerte')
plt.xlabel(r'$I/$A')
plt.ylabel(r'$B/$mT')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/hysterese.pdf')

plt.clf()
