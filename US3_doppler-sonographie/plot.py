import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat
from matrix2latex import matrix2latex

def f(x, a, b):
   return a*x + b


alpha_15=80.06
alpha_30=70.53
alpha_60=54.74

nu_0=2*1e6 #Hz   ->Sonde hat 2MHz
c_L=1800 #Strömungsgeschwindigkeit in der Flüssigkeit
c_P=2700 #Strömungsgeschwindigkeit im Prisma

#dickes Rohr
d_dick=16 #mm

rpm, deltanu_15, deltanu_30, deltanu_60  = np.genfromtxt('data/dick.txt', unpack=True)

v_15=deltanu_15*c_L/(2*nu_0*np.cos(alpha_15*np.pi/180))
v_30=deltanu_30*c_L/(2*nu_0*np.cos(alpha_30*np.pi/180))
v_60=deltanu_60*c_L/(2*nu_0*np.cos(alpha_60*np.pi/180))

plt.plot(v_15, deltanu_15/np.cos(np.pi/180*alpha_15), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dick_15.pdf')
plt.clf()

plt.plot(v_30, deltanu_30/np.cos(np.pi/180*alpha_30), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dick_30.pdf')
plt.clf()

plt.plot(v_60, deltanu_60/np.cos(np.pi/180*alpha_60), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dick_60.pdf')
plt.clf()

hr = ['U/min','$\Delta\nu_{15}/$Hz','$\Delta\nu_{30}/$Hz','$\Delta\nu_{60}/$Hz',
'$v_{15}/$\frac{m}{s}','$v_{30}/$\frac{m}{s}','$v_{60}/$\frac{m}{s}']
m = np.zeros((6, 7))
m[:,0] = rpm
m[:,1] = deltanu_15
m[:,2] = deltanu_30
m[:,3] = deltanu_60
m[:,4] = v_15
m[:,5] = v_30
m[:,6] = v_60
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

#mittleres Rohr
d_mittel=10 #mm

rpm, deltanu_15, deltanu_30, deltanu_60  = np.genfromtxt('data/mittel.txt', unpack=True)

v_15=deltanu_15*c_L/(2*nu_0*np.cos(alpha_15*np.pi/180))
v_30=deltanu_30*c_L/(2*nu_0*np.cos(alpha_30*np.pi/180))
v_60=deltanu_60*c_L/(2*nu_0*np.cos(alpha_60*np.pi/180))

plt.plot(v_15, deltanu_15/np.cos(np.pi/180*alpha_15), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/mittel_15.pdf')
plt.clf()

plt.plot(v_30, deltanu_30/np.cos(np.pi/180*alpha_30), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/mittel_30.pdf')
plt.clf()

plt.plot(v_60, deltanu_60/np.cos(np.pi/180*alpha_60), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/mittel_60.pdf')
plt.clf()

hr = ['U/min','$\Delta\nu_{15}/$Hz','$\Delta\nu_{30}/$Hz','$\Delta\nu_{60}/$Hz',
'$v_{15}/$\frac{m}{s}','$v_{30}/$\frac{m}{s}','$v_{60}/$\frac{m}{s}']
m = np.zeros((6, 7))
m[:,0] = rpm
m[:,1] = deltanu_15
m[:,2] = deltanu_30
m[:,3] = deltanu_60
m[:,4] = v_15
m[:,5] = v_30
m[:,6] = v_60
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)

#dünnes Rohr
d_dünn=7 #mm

rpm, deltanu_15, deltanu_30, deltanu_60  = np.genfromtxt('data/dünn.txt', unpack=True)

v_15=deltanu_15*c_L/(2*nu_0*np.cos(alpha_15*np.pi/180))
v_30=deltanu_30*c_L/(2*nu_0*np.cos(alpha_30*np.pi/180))
v_60=deltanu_60*c_L/(2*nu_0*np.cos(alpha_60*np.pi/180))

plt.plot(v_15, deltanu_15/np.cos(np.pi/180*alpha_15), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dünn_15.pdf')
plt.clf()

plt.plot(v_30, deltanu_30/np.cos(np.pi/180*alpha_30), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dünn_30.pdf')
plt.clf()

plt.plot(v_60, deltanu_60/np.cos(np.pi/180*alpha_60), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/dünn_60.pdf')
plt.clf()

hr = ['U/min','$\Delta\nu_{15}/$Hz','$\Delta\nu_{30}/$Hz','$\Delta\nu_{60}/$Hz',
'$v_{15}/$\frac{m}{s}','$v_{30}/$\frac{m}{s}','$v_{60}/$\frac{m}{s}']
m = np.zeros((6, 7))
m[:,0] = rpm
m[:,1] = deltanu_15
m[:,2] = deltanu_30
m[:,3] = deltanu_60
m[:,4] = v_15
m[:,5] = v_30
m[:,6] = v_60
t=matrix2latex(m, headerRow=hr, format='%.2f')
print(t)
