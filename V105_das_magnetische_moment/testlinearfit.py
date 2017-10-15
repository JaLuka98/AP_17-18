import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit

def f(x, a, b):
    return a*x+b


r = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
I1arr = np.array([1.7,1.67,1.7,1.7,1.68])
I2arr = np.array([1.95,1.95,1.93,1.9,1.9])
I3arr = np.array([2.2,2.2,2.17,2.18,2.14])
I1 = ufloat(np.mean(I1arr), np.std(I1arr))
I2 = ufloat(np.mean(I2arr), np.std(I2arr))
I = np.array([unp.nominal_values(I1), unp.nominal_values(I2)])
Idata = np.concatenate((I1arr, I2arr, I3arr))
weights = np.array([unp.std_devs(I1), unp.std_devs(I2)])
#params = np.polyfit(r, I, deg=1, rcond=None, full=False, w=weights)
params, pcov = curve_fit(f, r, Idata)
print(Idata)
print(weights)
print(params)
plt.plot(r, params[0]*r+params[1], '-', markersize = 10, mew = 20)
plt.plot(r, Idata, '.')
plt.savefig('test.pdf')
