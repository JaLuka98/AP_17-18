import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


f_1=150 #Brennweite der Bekannten Linse in mm; benutzt für erste Messreihe und Bessel
f_2=-50  #Brennweite einer Linse bei Abbe in mm
f_3=50  #Brennweite einer Linse bei Abbe in mm#

G=3 #Gegenstandsgröße in cm
