import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def cosfit(phi, U_m, U_0, delta):
    return U_m*np.cos(phi+delta)+U_0


#f=1000 Hz
#U_0=10 mV
#U_ss=11.4 mv   Amplitude des Rauschens
#preamplifier: Gain=10
#filter:Q=2     frequency range=300-1000Hz
#Lock_in Detector: Gain=5
#Tiefpass: tau=1-3s     Gain=10

phi, U, U_R = np.genfromtxt('data/phase.txt', unpack=True)
phi = phi*2*np.pi/360  # phi nun in rad

# Das ungerauschte Signal

params1, covariance_matrix1 = optimize.curve_fit(cosfit, phi, U)

errors1 = np.sqrt(np.diag(covariance_matrix1))

print('Die Werte für den Fit des ungerauschten Signals betragen')

print('U_m =', params1[0], '+-', errors1[0])
print('U_0 =', params1[1], '+-', errors1[1])
print('delta', params1[2], '+-', errors1[2])

philinspace = np.linspace(-np.pi/8, 2*np.pi+np.pi/8, 100)

plt.plot(phi, U, 'rx', label='Messwerte')
plt.plot(philinspace, cosfit(philinspace, *params1), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$\phi$/rad')
plt.ylabel(r'$U_\mathrm{out,1}$/\,V')
plt.axis((-np.pi/8, 2*np.pi+np.pi/8, -5.5, 5.5))

plt.xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi/4, np.pi, 5*np.pi / 4, 3*np.pi / 2, 7*np.pi / 4, 2*np.pi],
           [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$", r"$\frac{5}{4}\pi$", r"$\frac{3}{2}\pi$", r"$\frac{7}{4}\pi$", r"$2 \pi$"])

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/ohnerauschen.pdf')

plt.clf()

# Das gerauschte Signal

params2, covariance_matrix2 = optimize.curve_fit(cosfit, phi, U_R)

errors2 = np.sqrt(np.diag(covariance_matrix2))

print('Die Werte für den Fit des gerauschten Signals betragen')

print('U_m =', params2[0], '+-', errors2[0])
print('U_0 =', params2[1], '+-', errors2[1])
print('delta', params2[2], '+-', errors2[2])

plt.plot(phi, U_R, 'rx', label='Messwerte')
plt.plot(philinspace, cosfit(philinspace, *params2), 'k-', label='Ausgleichsfunktion')
plt.xlabel(r'$\phi$/rad')
plt.ylabel(r'$U_\mathrm{out,2}$/\,V')
plt.axis((-np.pi/8, 2*np.pi+np.pi/8, -5.5, 5.5))
plt.xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi/4, np.pi, 5*np.pi / 4, 3*np.pi / 2, 7*np.pi / 4, 2*np.pi],
           [r"$0$", r"$\frac{1}{4}\pi$", r"$\frac{1}{2}\pi$", r"$\frac{3}{4}\pi$", r"$\pi$", r"$\frac{5}{4}\pi$", r"$\frac{3}{2}\pi$", r"$\frac{7}{4}\pi$", r"$2 \pi$"])
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('build/mitrauschen.pdf')
