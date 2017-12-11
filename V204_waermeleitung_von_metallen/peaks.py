import peakutils

import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


def expfit(t, A, M, c):
    return M-A*np.exp(-c*t)


# Dynamisch200

t, T7, T8 = np.genfromtxt('data/dynamisch200.txt', unpack=True)

indexesmax = peakutils.indexes(T7, thres=0.02/max(T7), min_dist=50)
indexesmin = peakutils.indexes(-T7, thres=0.02/max(-T7), min_dist=50)

params, covariance_matrix = optimize.curve_fit(expfit, indexesmax, T7[indexesmax], p0=[10,45,0.01])

print(params)

errors = np.sqrt(np.diag(covariance_matrix))

plt.plot(t, T7, 'b-', label='Messwerte')
plt.plot(indexesmax, T7[indexesmax], 'rx', label='maxima')
plt.plot(indexesmin, T7[indexesmin], 'rx', label='minima')
plt.plot(indexesmax, expfit(indexesmax, *params), 'k-', label='fit über die Peaks')
plt.plot(t, np.abs(T7-expfit(t, *params)), 'g-', label='der Cosinus')
plt.legend()
plt.grid()
plt.savefig('build/test.pdf')

# print(indexesmax)
# print(indexesmin)
