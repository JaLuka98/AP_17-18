import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit


def f(x, a, b):
    return a*x+b


x = np.array([0,0,0,1,1,1,2,2,2])
y = np.array([0,0,0,0,4,2,2,2,2])
params, pcov = curve_fit(f, x, y)
plt.plot(x, params[0]*x+params[1], '-', markersize = 10, mew = 20)
params, pcov = curve_fit(f, [0,1,2],[0,2,2])
plt.plot(x, params[0]*x+params[1], '-', markersize = 10, mew = 20)
plt.plot(x, y, '.')
plt.savefig('testver.pdf')
