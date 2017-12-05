import numpy as np
import scipy.optimize
import uncertainties as unc
import uncertainties.unumpy as unp
from scipy import optimize
import matplotlib.pyplot as plt
from uncertainties import ufloat


#f=1000 Hz
#U_0=10 mV
#U_ss=11.4 mv   Amplitude des Rauschens
#preamlifier: Gain=10
#filter:Q=2     frequency range=300-1000Hz
#Lock_in Detector: Gain=5
#Tiefpass: tao=1-3s     Gain=10
