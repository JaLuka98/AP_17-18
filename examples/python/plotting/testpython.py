import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def f(x, a, b):
    return a*np.exp(b*x)


x = np.linspace(0, 3, 50)
y = np.zeros(100)
for number in x:
    y = 1*np.exp(x)+np.random.normal(0, 1, 50)

params, covariance_matrix = curve_fit(f, x, y)

errors = np.sqrt(np.diag(covariance_matrix))
print('a =', params[0], '+-', errors[0])
print('b =', params[1], '+-', errors[1])

plt.plot(x, y, 'k.', label='Messwerte')
plt.plot(x, f(x, *params), 'r-', label='fit')
plt.legend()
plt.tight_layout()
plt.savefig('test.pdf')
plt.clf()
plt.plot(x, y, 'k.', label='Messwerte')
plt.plot(x, f(x, *params), 'r-', label='fit')
plt.legend()
plt.yscale('log')
plt.savefig('test2.pdf')
