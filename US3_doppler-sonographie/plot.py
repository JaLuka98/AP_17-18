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

#Mittelwerte

v_alle=np.vstack([v_15, v_30, v_60])
v_alle=np.absolute(v_alle)
v_mitteldick=np.mean(v_alle, 0)
v_mittelfehlerdick=np.std(v_alle, 0)

print(v_mitteldick)
print(v_mittelfehlerdick)


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

#Mittelwerte

v_alle=np.vstack([v_15, v_30, v_60])
v_alle=np.absolute(v_alle)
v_mittelmittel=np.mean(v_alle, 0)
v_mittelfehlermittel=np.std(v_alle, 0)

print(v_mittelmittel)
print(v_mittelfehlermittel)

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

#duennes Rohr
d_duenn=7 #mm

rpm, deltanu_15, deltanu_30, deltanu_60  = np.genfromtxt('data/duenn.txt', unpack=True)

v_15=deltanu_15*c_L/(2*nu_0*np.cos(alpha_15*np.pi/180))
v_30=deltanu_30*c_L/(2*nu_0*np.cos(alpha_30*np.pi/180))
v_60=deltanu_60*c_L/(2*nu_0*np.cos(alpha_60*np.pi/180))

v_alle=np.vstack([v_15, v_30, v_60])
v_alle=np.absolute(v_alle)
v_mitteldünn=np.mean(v_alle, 0)
v_mittelfehlerdünn=np.std(v_alle, 0)

print(v_mitteldünn)
print(v_mittelfehlerdünn)

plt.plot(v_15, deltanu_15/np.cos(np.pi/180*alpha_15), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/duenn_15.pdf')
plt.clf()

plt.plot(v_30, deltanu_30/np.cos(np.pi/180*alpha_30), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/duenn_30.pdf')
plt.clf()

plt.plot(v_60, deltanu_60/np.cos(np.pi/180*alpha_60), 'rx', mew=0.5, label='Messwerte')
plt.xlabel(r'$v/$(m/s)')
plt.ylabel(r'$(\Delta \nu / \cos(\alpha))/$Hz')
plt.tight_layout()
plt.legend()
plt.grid()
plt.savefig('build/duenn_60.pdf')
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

hr = ['U/min','$\nu_{15}\bar/$Hz','$\Delta\nu_{15}\bar/$Hz',
'$\nu_{30}\bar/$Hz','$\Delta\nu_{30}\bar/$Hz',
'$\nu_{60}\bar/$Hz','$\Delta\nu_{60}\bar/$Hz']
m = np.zeros((6, 7))
m[:,0] = rpm
m[:,1] = v_mitteldick
m[:,2] = v_mittelfehlerdick
m[:,3] = v_mittelmittel
m[:,4] = v_mittelfehlermittel
m[:,5] = v_mitteldünn
m[:,6] = v_mittelfehlerdünn
t=matrix2latex(m, headerRow=hr, format='%.3f')
print(t)

# stroemungsprofil

t, nu, sigma = np.genfromtxt('data/stroemung70.txt', unpack=True)
nu[0:5] *= c_P/(2*nu_0*np.cos(alpha_15*np.pi/180))
nu[5:] *= (12.28/t[5:]*c_P+((1-12.28/t[5:])*c_L))/(2*nu_0*np.cos(alpha_15*np.pi/180))
v = nu  # Umrechnung in Geschwindikeit erfolgt
plt.subplot(2, 1, 1)
plt.plot(t, np.abs(v), 'x', color='firebrick', mew=0.5, label=r'Mittlere Strömungsgeschwindigkeit')
plt.xlabel(r'$t/$µs')
plt.ylabel(r'$|\bar{v}|/$(m/s)')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(t, sigma, 'rx', color = 'orchid', mew=0.5, label=r'Streuintensität')
plt.xlabel(r'$t/$µs')
plt.ylabel(r'$\mathrm{Streuintensität}/\%$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=1, fancybox=True)
plt.grid()
plt.tight_layout()
plt.savefig('build/stroemung70.pdf')
plt.clf()

t, nu, sigma = np.genfromtxt('data/stroemung45.txt', unpack=True)
nu[0:5] *= c_P/(2*nu_0*np.cos(alpha_15*np.pi/180))
nu[5:] *= (12.28/t[5:]*c_P+((1-12.28/t[5:])*c_L))/(2*nu_0*np.cos(alpha_15*np.pi/180))
v = nu  # Umrechnung in Geschwindikeit erfolgt
plt.subplot(2, 1, 1)
plt.plot(t, np.abs(v), 'x', color='firebrick', mew=0.5, label=r'Mittlere Strömungsgeschwindigkeit')
plt.xlabel(r'$t/$µs')
plt.ylabel(r'$|\bar{v}|/$(m/s)')
plt.legend(loc='upper center', bbox_to_anchor=(0.175, -0.15),
          ncol=1, fancybox=True)
plt.grid()
plt.subplot(2,1,2)
plt.plot(t, sigma, 'rx', color = 'orchid', mew=0.5, label=r'Streuintensität')
plt.xlabel(r'$t/$µs')
plt.ylabel(r'$\mathrm{Streuintensität}/\%$')
plt.legend(loc='upper center', bbox_to_anchor=(0.75, 1.425),
          ncol=1, fancybox=True)
plt.grid()
plt.tight_layout()
plt.savefig('build/stroemung45.pdf')
plt.clf()
