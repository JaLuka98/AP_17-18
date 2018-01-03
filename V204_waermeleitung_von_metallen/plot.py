import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


A1, A2, ln, t = np.genfromtxt('data/messing.txt', unpack=True)

A1_m = ufloat(np.sum(A1)/len(A1), np.std(A1))
A2_m = ufloat(np.sum(A2)/len(A2), np.std(A2))
ln_m = ufloat(np.sum(ln)/len(ln), np.std(ln))
t_m = ufloat(np.sum(t)/len(t), np.std(t))

k_messing = 8520*385*(0.03)**2/(2*t_m*ln_m)

print('Mittelwert zu A1:',A1_m)
print('Mittelwert zu A2:',A2_m)
print('Mittelwert zu ln:',ln_m)
print('Mittelwert zu t:',t_m)
print('Kappa für Messing:',k_messing)





A3, A4, t2 = np.genfromtxt('data/aluminium.txt', unpack=True)

ln_2=np.log(A3/A4)

A3_m = ufloat(np.sum(A3)/len(A3), np.std(A3))
A4_m = ufloat(np.sum(A4)/len(A4), np.std(A4))
ln2_m = ufloat(np.sum(ln_2)/len(ln_2), np.std(ln_2))
t2_m = ufloat(np.sum(t2)/len(t2), np.std(t2))

k_aluminium = 2800*830*(0.03)**2/(2*t2_m*ln2_m)

print('Mittelwert zu A3:',A3_m)
print('Mittelwert zu A4:',A4_m)
print('Mittelwert zu ln_2:',ln2_m)
print('Mittelwert zu t2:',t2_m)
print('Kappa für Aluminium:',k_aluminium)
print('ln_2:', ln_2)





A5, A6, t3 = np.genfromtxt('data/edelstahl.txt', unpack=True)

ln_3=np.log(A5/A6)

A5_m = ufloat(np.sum(A5)/len(A5), np.std(A5))
A6_m = ufloat(np.sum(A6)/len(A6), np.std(A6))
ln3_m = ufloat(np.sum(ln_3)/len(ln_3), np.std(ln_3))
t3_m = ufloat(np.sum(t3)/len(t3), np.std(t3))

k_edelstahl = 8000*400*(0.03)**2/(2*t3_m*ln3_m)

print('Mittelwert zu A5:',A5_m)
print('Mittelwert zu A6:',A6_m)
print('Mittelwert zu ln_3:',ln3_m)
print('Mittelwert zu t3:',t3_m)
print('Kappa für Edelstahl:', k_edelstahl)
print('ln_3:', ln_3)
