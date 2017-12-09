import numpy as np
import peakutils
import matplotlib.pyplot as plt

t, T7, T8 = np.genfromtxt('dynamisch200.txt', unpack=True)

indexesmax = peakutils.indexes(T7, thres=0.02/max(T7), min_dist=50)
indexesmin = peakutils.indexes(-T7, thres=0.02/max(-T7), min_dist=50)

plt.plot(t, T7, 'b-', label='Messwerte')
plt.plot(indexesmax, T7[indexesmax], 'rx')
plt.plot(indexesmin, T7[indexesmin], 'rx')
#plt.show()

print(indexesmax)
print(indexesmin)
