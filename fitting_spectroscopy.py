'''----------------------------------------Imports----------------------------------------
'''

import sys
import re
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import math as m
from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.stats as stats

from scipy.optimize import curve_fit
from scipy.optimize.optimize import OptimizeWarning
import scipy.constants as const
import pickle
from sympy.physics.wigner import wigner_6j
from sympy.physics.wigner import wigner_3j
from datetime import datetime
import copy
import warnings
from pathlib import Path
import gc

t = u"\u00b0"


'''----------------------------------------Flags and Options----------------------------------------
'''


### Command line takes as arguments
###		1) ./path/to/file.txt
###		2) LN2 density (in torr)
###		3) Fit Used
### Example: python fitting_spectroscopy.py ./Noah/run1/Noah_run1_145C_#1.txt 74.9 3

'''	len = 2, just a file
	len = 3, N2 density too (density is [2])
	len = 4, externally selecting fit (fitSelect is [3])
	len = 5, iterating initials (the variable is [4])
'''

### Choose a fitting
	# 0 - Integral
	# 1 - Simple Lorentzian
	# 2 - Select Asymmetric sum
	# 3 - Select pV, unbound eta
	# 4 - Select pV, limited eta
	# 5 - Select pV, defined eta
	# 6 - Full Lorentzian
	# 7 - Full pV, unbound eta
	# 8 - Full pV, limited eta
	# 9 - Full pV, defined eta
fitSelection = 3
updates = 1
if len(sys.argv) > 3:
	fitSelection = int(sys.argv[3])
	updates = 0

if len(sys.argv) > 4:
	wild1 = float(sys.argv[4].split("to")[0])
	wild2 = float(sys.argv[4].split("to")[1])
else:
	wild1 = 100
	wild2 = 100


'''
					 0     1     2     3     4     5       6     7     8     9     10    11      12   13   14
Rb_lorentzian        c0,   c1,   c2,   c3,   m,    b
KD1D2_lorentzian     D1c0, D1c1, D1c2, D1c3, D2c0, D2c1,   D2c2, D2c3, m,    b

RbD1_pseudoVoight    c0,   c1,   c2,   fl,   fg,   eta,    m,    b,    definedEta
KD1D2_pseudoVoight   D1c0, D1c1, D1c2, D1fl, D1fg, D1_eta, D2c0, D2c1, D2c2, D2fl, D2fg, D2_eta, m,   b,   definedEta
'''

### Choose Results Printed (0=False, 1=True)
includeAlkaliDensityRatio = 0
includeHeliumDensity = 0

### Select Plots Displayed (0=False, 1=True)
includePlots = 1
annotated = 1
plot_diagnostic = 0
plot_eachLine = 0
saveFigures = 0

### Slicing options
allowedError = 1 ### Allowed error for any given data point (0.00 to 1.00)
wingPercentCut = 0 ### Cut a percentage off wings (0.10 = 10%, 0.25 = 25%, etc)

### Initial Values
in_FWHM = 100 ### The initial guess at FWHM (~100 for targets, ~10 for kappas)
manyTries = 100000000 ### How many attempts to fit the line
conSet = "VN1" ### Choose the set of constants used for calculating helium-3 density # conSet options:	JD1, JD2, VN1, VN2, VN3, TA1, TA1_den (for line shift analysis, not FWHM)

errorFlag = False

fitUsed = ["Integral", "Lorentzian_simple", "Lorentzian_asym", "pV_unbound_selectTransitions", "pV_limitedEta_selectTransitions", "pV_definedEta_selectTransitions", "Lorentzian_full", "pV_unbound_allTransitions", "pV_limitedEta_allTransitions", "pV_definedEta_allTransitions"][fitSelection]
included_RbD1, included_RbD2, included_KD1, included_KD2 = (False for _ in range(4))
savePath = Path(".", "__AllData", "ButterballOnly")
homePath = Path.cwd()
try:
	os.makedirs(savePath)
except:
	pass

warnings.simplefilter("ignore", category=OptimizeWarning)
warnings.simplefilter("error", category=OptimizeWarning)

'''----------------------------------------Functions----------------------------------------
'''


################## I/O

### Notes "Pull it out the brine!"
	# Purpose: Import a serialized file
	# Takes: string, name of file
	# Returns: array
def importAndUnpickle(fileIn):
	print("Unpickling %s." % fileIn)
	with open(fileIn, 'rb') as x:
		temp = pickle.load(x)
	print("\tUnpickling complete.")
	return temp

### Notes "Dunk it in the brine!"
	# Purpose: Export a serialized file (read by "run_multiSpec.py")
	# Takes: n (thing being pickled), line (RbD1, RbD2, KD1D2)
	# Exports: Pickled file (usually of residuals) labelled with the line it's for
def export_toPickle(n, note):
	fileName = f"exportDict{note}.pickle"
	with open(fileName, 'wb') as e:
		pickle.dump(n, e)

################## FITS


#------------Simple

# Notes "Lorentzian from Romalis paper (Rb)"
	# Purpose: Find Lorentzian distribuion in Rb line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line), c3 (FWHM)
	# Returns: Value for Lorentzian distribuion in Rb line at point x
def Rb_lorentzian_simple( x, c0, c1, c2, c3, m, b):
	return (c0*c1*((x-c2)/2)) / ((x-c2)**2 + (c3/2)**2) + m*x + b

# Notes "Lorentzian from Romalis paper (K)"
	# Purpose: Find Lorentzian distribuion in Rb line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line), c3 (FWHM)
	# Returns: Value for Lorentzian distribuion in Rb line at point x
def K_lorentzian_simple( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
	D1 = (D1c0*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)
	D2 = (D2c0*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)
	return (D1 + D2 + m*x + b)

#------------Old and Busted

# Notes "Lorentzians from Jaideep's Thesis (Rb)"
	# Purpose: Find Lorentzian distribuion in Rb line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line), c3 (FWHM)
	# Returns: Value for Lorentzian distribuion in Rb line at point x
def Rb_lorentzian_asymSimple( x, c0, c1, c2, c3, m, b):
	del1 = x - c2 + 1.264887
	del2 = x - c2 - 1.770884
	del3 = x - c2 + 2.563005
	del4 = x - c2 - 4.271676
	part1 = (1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2)
	part2 = (1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2)
	part3 = (1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2)
	part4 = (1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2)
	return (0.7217*c0*( (7/12)*part1 + (5/12)*part2 ) + 0.2783*c0*( (5/8)*part3 + (3/8)*part4 ) + m*x + b )

# Notes "Lorentzians from Jaideep's Thesis (K)"
	# Purpose: Find Lorentzian distribuion in K line at point x
	# Takes: x (position), D1c0 (-[A]*len*e*c*f/2), D1c1 (interaction time), D1c2 (center of line), D1c3 (FWHM), corresponding values for D2
	# Returns: Value for Lorentzian distribuion in Rb line at point x
def K_lorentzian_asymSimple( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
	D1 = D1c0*(1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)
	D2 = D2c0*(1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)
	return (D1 + D2 + m*x + b)

#------------New Hotness

# Notes "pV using shifts, weights from Jaideep's Thesis (K)"
	# Purpose: Find PseudoVoight distribuion in K line at point x
	# Takes: x (position), D1c0 (-[A]*len*e*c*f/2), D1c1 (interaction time), D1c2, (center of line), D1fl (FWHM for Lorentzian), D1fg (FWHM for gaussian)
			# similar values for the D2 line, m (linear slope), b (linear intercept)
	# Returns: Value for PseudoVoight distribuion in K line at point x
def K_pseudoVoight_asymSimple( x, D1c0, D1c1, D1c2, D1fl, D1fg, D1_eta, D2c0, D2c1, D2c2, D2fl, D2fg, D2_eta, m, b, definedEta):
	D1gam = D1fl / 2
	D2gam = D2fl / 2
	D1sig = D1fg / np.sqrt(8*np.log(2))
	D2sig = D2fg / np.sqrt(8*np.log(2))

	if definedEta:
		D1f = ( (D1fg**5) + 2.69269*(D1fg**4)*(D1fl) + 2.42843*(D1fg**3)*(D1fl**2) + 4.47163*(D1fg**2)*(D1fl**3) + 0.07842*(D1fg)*(D1fl**4) + (D1fl**5) )**(0.2)
		D1_eta = 1.36603*(D1fl/D1f) - 0.47719*((D1fl/D1f)**2) + 0.11116*((D1fl/D1f)**3)
		if included_KD2:
			D2f = ( (D2fg**5) + 2.69269*(D2fg**4)*(D2fl) + 2.42843*(D2fg**3)*(D2fl**2) + 4.47163*(D2fg**2)*(D2fl**3) + 0.07842*(D2fg)*(D2fl**4) + (D2fl**5) )**(0.2)
			D2_eta = 1.36603*(D2fl/D2f) - 0.47719*((D2fl/D2f)**2) + 0.11116*((D2fl/D2f)**3)

	D1 = D1_eta*((1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1gam)**2)) + (1-D1_eta)*( (1. / np.sqrt( 2*np.pi*D1sig*D1sig) )*( np.exp( -0.5*((x-D1c2)/D1sig)**2 ) ) )
	D2 = 0
	if included_KD2:
		D2 = D2_eta*((1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2gam)**2)) + (1-D2_eta)*( (1. / np.sqrt( 2*np.pi*D2sig*D2sig) )*( np.exp( -0.5*((x-D2c2)/D2sig)**2 ) ) )
	return (D1c0*D1 + D2c0*D2 + m*x + b)






#------------Let's overthink it

### Notes "Strength of a transmission between ground and excited hyperfine states (Rb)"
	# Purpose: Find a given tranmission strength
	# Takes: momentum values f_g (F for ground), f_e (F for excited state), j_e (J for excited state), I (total nuclear spin)
	# Exports: float
def HF_tranStrength_Rb(f_g, f_e, j_e, I):
	return (2*f_e + 1)*(2*(1/2) + 1)*(wigner_6j(1/2, j_e, 1, f_e, f_g, I)**2)

### Notes "Frequency shift from weighted central frequency (Rb)"
	# Purpose: Calculate total frequency shift due to isotope and offset of excited and ground states from their average for all isotopes 
	# Takes: f_g (total ang momentum, F, for ground), f_e (total ang momentum, F, for excited state), isotope number (85 or 87)
	# Exports: float
def HF_tranShift_Rb(f_g, f_e, isotope, transition): ### all freq in GHz
	isoShift, g, e = ([] for _ in range(3))
	if isotope == "85":
		isoShift = [m.nan, -0.0216211269726045, -0.0217339776572771] ## Shift from average peak for [nan,D1,D2] lines
		g = [m.nan, m.nan, -1.7708439228, 1.2648885163] ## hyperfine shift where Rb85_g[F] for energy level F
		e = [[m.nan], [m.nan, m.nan, -0.210923, 0.150659], [m.nan, -0.113208, -0.083835, -0.20435, 0.100205]]
	elif isotope == "87":
		isoShift = [m.nan, 0.0560688729747199, 0.0563615223509260]
		g = [m.nan, -4.271676631815181, 2.563005979089109]
		e = [[m.nan], [m.nan, -0.50906, 0.30544], [-0.3020738, -0.2298518, -0.0729112, 0.1937407]]
	else:
		print("NOT A VALID ISOTOPE")
		return 0
	return (isoShift[int(transition)] + e[int(transition)][int(f_e)] + g[int(f_g)])

# Notes "Full Lorentzian, summed. Terms weighted by isotope, hyperfine transition (RbD1 only)"
	# Purpose: Find Lorentzian distribuion in RbD1 line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line), c3 (FWHM), m (linear slope), b (linear intercept)
	# Returns: Value for Lorentzian distribuion in RbD1 line at point x
def RbD1_lorentzian_full( x, c0, c1, c2, c3, m, b):
	line = 1
	J = 1/2
	### Rb85 ###
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(2,4):
		for f_e in range(2,4):
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "85", line)))
	for idx, val in enumerate(temp_f):
		temp.append( ( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (c3/2)**2) ) ) )
	Rb85 = sum(temp)
	### Rb87 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		for f_e in range(1,3):
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "87", line)))
	for idx, val in enumerate(temp_f):
		temp.append( ( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (c3/2)**2) ) ) )
	Rb87 = sum(temp)
	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

# Notes "Full Lorentzian, summed. Terms weighted by isotope, hyperfine transition (RbD2 only)"
	# Purpose: Find Lorentzian distribuion in RbD2 line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line), c3 (FWHM), m (linear slope), b (linear intercept)
	# Returns: Value for Lorentzian distribuion in RbD2 line at point x
def RbD2_lorentzian_full( x, c0, c1, c2, c3, m, b):
	line = 2
	J = 3/2

	### Rb85
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(2,4):
		f_e = int(f_g-1)
		while f_e <= f_g+1:
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "85", line)))
			f_e += 1
	for idx, val in enumerate(temp_f):
		temp.append( val*( (1 + (0.664*2*np.pi*c1*(x - c2 + temp_d[idx]))) / ((x - c2 + temp_d[idx])**2 + (c3/2)**2) ) )
	Rb85 = sum(temp)
	### Rb87
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		f_e = int(f_g-1)
		while f_e <= f_g+1:
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "87", line)))
			f_e += 1
	for idx, val in enumerate(temp_f):
		temp.append( val*( (1 + (0.664*2*np.pi*c1*(x - c2 + temp_d[idx]))) / ((x - c2 + temp_d[idx])**2 + (c3/2)**2) ) )
	Rb87 = sum(temp)
	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

### Notes "Strength of a transmission between ground and excited hyperfine states (K)"
	# Purpose: Find a given tranmission strength
	# Takes: f_g (total ang momentum, F, for ground), f_e (total ang momentum, F, for excited state),
			# j_e (total electron angular momentum, J, for excited state), I (total nuclear spin)
			# mf_g (projection of F for ground state), and mf_e (projection of F for excited state)
	# Exports: float
	# Add'l Note: [s_g = 1/2, j_g = 1/2, l_g = 0, l_e = 1] also (F = I + J) and (J = L + S)
def HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e):
	return (2 * (2*j_e+1) * (2*f_g+1) * (2*f_e+1))*(( wigner_6j(1, j_e, 1/2, 1/2, 0, 1)*wigner_6j(j_e, f_e, I, f_g, 1/2, 1)*wigner_3j(f_g, 1, f_e, mf_g, 0, -1*mf_e) )**2)

### Notes "Frequency shift from weighted central frequency (Rb)"
	# Purpose: Calculate total frequency shift due to isotope and offset of excited and ground states from their average for all isotopes 
	# Takes: f_g (total ang momentum, F, for ground), f_e (total ang momentum, F, for excited state), isotope number (39, 40, or 41), transition (1 or 2)
	# Exports: float
def HF_tranShift_K(f_g, f_e, isotope, transition): ### all freq in GHz
	isoShift, g, e = ([] for _ in range(3))
	if isotope == "39":
		isoShift = [m.nan, -0.0158635801636, -0.0159101306926]
		g = [m.nan, -0.2886, 0.1731]
		e = [[m.nan], [m.nan, -0.0347, 0.0208], [-0.0194, -0.0161, -0.0067, 0.0144]]
	elif isotope == "40":
		f_g = int(11/2 - f_g)
		f_e = int(11/2 - f_e)
		isoShift = [m.nan, 0.1097734198556, 0.1101098693325]
		g = [m.nan, -0.5715, 0.7143]
		e = [[m.nan], [m.nan, -0.069, 0.0863], [-0.0464, -0.0023, 0.031, 0.0552]]
	elif isotope == "41":
		isoShift = [m.nan, 0.2196254198207, 0.2202698693145]
		g = [m.nan, -0.1588, 0.0953]
		e = [[m.nan], [m.nan, -0.0191, 0.0114], [-0.0084, -0.0084, -0.005, 0.0084]]
	else:
		print("NOT A VALID ISOTOPE")
		return 0
	return isoShift[int(transition)] + e[int(transition)][int(f_e)] + g[int(f_g)]

# Notes "Full Lorentzian, summed. Terms weighted by isotope, hyperfine transition (KD1D2 only)"
	# Purpose: Find Lorentzian distribuion in RbD1 line at point x
	# Takes: x (position), D1c0 (-[A]*len*e*c*f/2), D1c1 (interaction time), D1c2, (center of line), D1c3 (FWHM),
			#similar values for the D2 line, m (linear slope), b (linear intercept)
	# Returns: Value for Lorentzian distribuion in KD1 and KD2 line at point x
def KD1D2_lorentzian_full( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
	bothLines = []

	### K39 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3): ### --------- D1 Line
		for f_e in range(1, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "39", 1)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( (1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx]) ))) / ((x - D1c2 + float(temp_d[idx]) )**2 + (D1c3/2)**2) ) ) )
	bothLines.append(sum(temp))
	for f_g in range(1,3): ### --------- D2 Line
		for f_e in range(0, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "39", 2)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( (1 + (0.664*2*np.pi*D2c1*(x - D2c2 + float(temp_d[idx]) ))) / ((x - D2c2 + float(temp_d[idx]) )**2 + (D2c3/2)**2) ) ) )
	bothLines.append(sum(temp))

	### K40 ###
	I = 4
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	f_g = 7/2
	while f_g < 5: ### --------- D1 Line
		f_e = 3.5
		while (f_e <= 4.5):
			if ( (np.round(f_e, 1) == np.round(f_g, 1)) or (abs(np.round(f_e, 1)) == np.round(f_g+1, 1)) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e))) ### HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e)
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "40", 1)))
						mf_e += 1
					mf_g += 1
			f_e += 1
		f_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.000117*( (val)*( (1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx]) ))) / ((x - D1c2 + float(temp_d[idx]) )**2 + (D1c3/2)**2) ) ) )
	bothLines.append(sum(temp))
	while f_g < 5: ### --------- D2 Line
		f_e = 4.5
		while (f_e <= 5.5):
			if ( (np.round(f_e, 1) == np.round(f_g, 1)) or (abs(np.round(f_e, 1)) == np.round(f_g+1, 1)) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e))) ### HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e)
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "40", 2)))
						mf_e += 1
					mf_g += 1
			f_e += 1
		f_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( (1 + (0.664*2*np.pi*D2c1*(x - D2c2 + float(temp_d[idx]) ))) / ((x - D2c2 + float(temp_d[idx]) )**2 + (D2c3/2)**2) ) ) )
	bothLines.append(sum(temp))

	### K41 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		for f_e in range(1, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "41", 1)))
						mf_e += 1
					mf_g += 1 ### --------- D1 Line
	for idx, val in enumerate(temp_f):
		temp.append( 0.000117*( (val)*( (1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx]) ))) / ((x - D1c2 + float(temp_d[idx]) )**2 + (D1c3/2)**2) ) ) )
	bothLines.append(sum(temp))
	for f_g in range(1,3): ### --------- D2 Line
		for f_e in range(0, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "41", 2)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( (1 + (0.664*2*np.pi*D2c1*(x - D2c2 + float(temp_d[idx]) ))) / ((x - D2c2 + float(temp_d[idx]) )**2 + (D2c3/2)**2) ) ) )
	bothLines.append(sum(temp))

	return ( D1c0*(bothLines[0]+bothLines[2]+bothLines[4]) + D2c0*(bothLines[1]+bothLines[3]+bothLines[5]) ) + m*x + b

#--------------Let's REALLY overthink it

# Notes "Full pV, summed. Terms weighted by isotope, hyperfine transition (RbD1 only)"
	# Purpose: Find PseudoVoight distribuion in Rb line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line),
		#fl (FWHM for Lorentzian), fg (FWHM for gaussian), eta (weighting function), m (linear slope), b (linear intercept)
	# Returns: Value for PseudoVoight distribuion in Rb line at point x
def RbD1_pseudoVoight_full( x, c0, c1, c2, fl, fg, eta, m, b, definedEta):
	line = 1
	J = 1/2

	gam = fl / 2
	sig = fg / np.sqrt(8*np.log(2))
	if definedEta:
		f = calc_pV_f(fl, fg, -1)
		eta = 1.36603*(fl/f) - 0.47719*((fl/f)**2) + 0.11116*((fl/f)**3)
		del f

	### Rb85 ###
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(2,4):
		for f_e in range(2,4):
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "85", line)))
			f_e += 1
				# Takes: f_g (total ang momentum, F, for ground), f_e (total ang momentum, F, for excited state), isotope number (85 or 87), transition (1 or 2)
	for idx, val in enumerate(temp_f):
		temp.append( eta*( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (gam)**2) ) ) + (1-eta)*( (1. / np.sqrt( 2*np.pi*sig*sig) )*( np.exp( -0.5*((x - c2 + float(temp_d[idx]))/sig)**2 ) ) ) )
	Rb85 = sum(temp)

	### Rb87 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		for f_e in range(1,3):
			temp_f.append(float(HF_tranStrength_Rb(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift_Rb(f_g, f_e, "87", line)))
			f_e += 1
	for idx, val in enumerate(temp_f):
		temp.append( eta*( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (gam)**2) ) ) + (1-eta)*( (1. / np.sqrt( 2*np.pi*sig*sig) )*( np.exp( -0.5*((x - c2 + float(temp_d[idx]))/sig)**2 ) ) ) )
	Rb87 = sum(temp)
	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

# Notes "Full pV, summed. Terms weighted by isotope, hyperfine transition (RbD2 only)"
	# Purpose: Find PseudoVoight distribuion in Rb line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line),
		#fl (FWHM for Lorentzian), fg (FWHM for gaussian), eta (weighting function), m (linear slope), b (linear intercept)
	# Returns: Value for PseudoVoight distribuion in Rb line at point x
def RbD2_pseudoVoight_full( x, c0, c1, c2, fl, fg, eta, m, b, definedEta):
	line = 2
	J = 3/2

	gam = fl / 2
	sig = fg / np.sqrt(8*np.log(2))
	if definedEta:
		f = calc_pV_f(fl, fg, -1)
		eta = 1.36603*(fl/f) - 0.47719*((fl/f)**2) + 0.11116*((fl/f)**3)
		del f

	### Rb85
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(2,4):
		f_e = int(f_g-1)
		while f_e <= f_g+1:
			temp_f.append(float(HF_tranStrength(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift(f_g, f_e, "Rb", "85", line)))
			f_e += 1
	for idx, val in enumerate(temp_f):
		temp.append( eta*( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (gam)**2) ) ) + (1-eta)*( (1. / np.sqrt( 2*np.pi*sig*sig) )*( np.exp( -0.5*((x - c2 + float(temp_d[idx]))/sig)**2 ) ) ) )
	Rb85 = sum(temp)

	### Rb87
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		f_e = int(f_g-1)
		while f_e <= f_g+1:
			temp_f.append(float(HF_tranStrength(f_g, f_e, J, I)))
			temp_d.append(float(HF_tranShift(f_g, f_e, "Rb", "87", line)))
			f_e += 1
	for idx, val in enumerate(temp_f):
		temp.append( eta*( (val)*( (1 + (0.664*2*np.pi*c1*(x - c2 + float(temp_d[idx]) ))) / ((x - c2 + float(temp_d[idx]) )**2 + (gam)**2) ) ) + (1-eta)*( (1. / np.sqrt( 2*np.pi*sig*sig) )*( np.exp( -0.5*((x - c2 + float(temp_d[idx]))/sig)**2 ) ) ) )
	Rb87 = sum(temp)

	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

# Notes "Full pV, summed. Terms weighted by isotope, hyperfine transition (K only)"
	# Purpose: Find PseudoVoight distribuion in K line at point x
	# Takes: x (position), c0 (-[A]*len*e*c*f/2), c1 (interaction time), c2, (center of line),
		#fl (FWHM for Lorentzian), fg (FWHM for gaussian), eta (weighting function), m (linear slope), b (linear intercept)
	# Returns: Value for PseudoVoight distribuion in K line at point x
def KD1D2_pseudoVoight_full( x, D1c0, D1c1, D1c2, D1fl, D1fg, D1_eta, D2c0, D2c1, D2c2, D2fl, D2fg, D2_eta, m, b, definedEta):
	bothLines = []

	D1gam = D1fl / 2
	D1sig = D1fg / np.sqrt(8*np.log(2))
	D2gam = D2fl / 2
	D2sig = D2fg / np.sqrt(8*np.log(2))

	if definedEta:
		D1f = calc_pV_f(D1fl, D1fg, -1)
		D1_eta = 1.36603*(D1fl/D1f) - 0.47719*((D1fl/D1f)**2) + 0.11116*((D1fl/D1f)**3)
		del D1f

		D2f = calc_pV_f(D2fl, D2fg, -1)
		D2_eta = 1.36603*(D2fl/D2f) - 0.47719*((D2fl/D2f)**2) + 0.11116*((D2fl/D2f)**3)
		del D2f

	else:
		pass

	### K39 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		for f_e in range(1, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "39", 1)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( D1_eta*((1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx])))) / ((x - D1c2 + float(temp_d[idx])**2 + (D1gam)**2)) + (1-D1_eta)*( (1. / np.sqrt( 2*np.pi*D1sig*D1sig) )*( np.exp( -0.5*((x - D1c2 + float(temp_d[idx])/D1sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))
	for f_g in range(1,3):
		for f_e in range(0, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "39", 2)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.932581*( (val)*( D2_eta*((1 + (0.664*2*np.pi*D2c1*(x - D2c2 + float(temp_d[idx])))) / ((x - D2c2 + float(temp_d[idx])**2 + (D2gam)**2)) + (1-D2_eta)*( (1. / np.sqrt( 2*np.pi*D2sig*D2sig) )*( np.exp( -0.5*((x - D2c2 + float(temp_d[idx])/D2sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))

	### K40 ###
	I = 4
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	f_g = 7/2
	while f_g < 5:
		f_e = 3.5
		while (f_e <= 4.5):
			if ( (np.round(f_e, 1) == np.round(f_g, 1)) or (abs(np.round(f_e, 1)) == np.round(f_g+1, 1)) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e))) ### HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e)
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "40", 1)))
						mf_e += 1
					mf_g += 1
			f_e += 1
		f_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.000117*( (val)*( D1_eta*((1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx])))) / ((x - D1c2 + float(temp_d[idx])**2 + (D1gam)**2)) + (1-D1_eta)*( (1. / np.sqrt( 2*np.pi*D1sig*D1sig) )*( np.exp( -0.5*((x - D1c2 + float(temp_d[idx])/D1sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))
	while f_g < 5:
		f_e = 4.5
		while (f_e <= 5.5):
			if ( (np.round(f_e, 1) == np.round(f_g, 1)) or (abs(np.round(f_e, 1)) == np.round(f_g+1, 1)) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e))) ### HF_tranStrength_K(f_g, f_e, j_e, I, mf_g, mf_e)
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "40", 2)))
						mf_e += 1
					mf_g += 1
			f_e += 1
		f_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.000117*( (val)*( D1_eta*((1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx])))) / ((x - D1c2 + float(temp_d[idx])**2 + (D1gam)**2)) + (1-D1_eta)*( (1. / np.sqrt( 2*np.pi*D1sig*D1sig) )*( np.exp( -0.5*((x - D1c2 + float(temp_d[idx])/D1sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))

	### K41 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	f_e = 0
	for f_g in range(1,3):
		for f_e in range(1, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 0.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "41", 1)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.067302*( (val)*( D1_eta*((1 + (0.664*2*np.pi*D1c1*(x - D1c2 + float(temp_d[idx])))) / ((x - D1c2 + float(temp_d[idx])**2 + (D1gam)**2)) + (1-D1_eta)*( (1. / np.sqrt( 2*np.pi*D1sig*D1sig) )*( np.exp( -0.5*((x - D1c2 + float(temp_d[idx])/D1sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))
	for f_g in range(1,3):
		for f_e in range(0, 3):
			if ( (f_e == f_g) or (abs(f_e) == f_g+1) ):
				mf_g = -1*f_g
				while (mf_g<=f_g):
					mf_e = -1*f_e
					while (mf_e<=f_e):
						temp_f.append(float(HF_tranStrength_K(f_g, f_e, 1.5, I, mf_g, mf_e)))
						temp_d.append(float(HF_tranShift_K(f_g, f_e, "41", 2)))
						mf_e += 1
					mf_g += 1
	for idx, val in enumerate(temp_f):
		temp.append( 0.067302*( (val)*( D2_eta*((1 + (0.664*2*np.pi*D2c1*(x - D2c2 + float(temp_d[idx])))) / ((x - D2c2 + float(temp_d[idx])**2 + (D2gam)**2)) + (1-D2_eta)*( (1. / np.sqrt( 2*np.pi*D2sig*D2sig) )*( np.exp( -0.5*((x - D2c2 + float(temp_d[idx])/D2sig)**2 ))))) ) ) )
	bothLines.append(sum(temp))

	return ( D1c0*(bothLines[0]+bothLines[2]+bothLines[4]) + D2c0*(bothLines[1]+bothLines[3]+bothLines[5]) ) + m*x + b

### 0-3, -1 for original
def calc_pV_f(l, g, mode):
		f = (g**5) + (2.69269*(g**4)*(l)) + (2.42843*(g**3)*(l**2)) + (4.47163*(g**2)*(l**3)) + (0.07842*(g)*(l**4)) + (l**5)
		if mode == -1:
			return f**2
		elif mopde == 0:
			return 0.5
		elif mode == 1:	
			return 0.5 + ( 0.5*(2**0.8)*((f-0.5) ) )
		elif mode == 2:
			return 0.5 + ( 0.5*(2**0.8)*((f-0.5) ) ) - ( (2/25)*(2**1.8)*((f-0.5)**2 ) )
		elif mode == 3:
			return 0.5 + ( 0.5*(2**0.8)*((f-0.5) ) ) - ( (2/25)*(2**1.8)*((f-0.5)**2 ) ) + ( (2/75)*(2**2.8)*((f-0.5)**3 ) )
		else:
			print(f"{mode} is not a proper choice")
			return m.nan


################## Alk Density

# Notes "Pure alkali density"
	# Purpose: Extrapolates an alkali ratio to operating temperature using the Nesmayanov method
	# Takes: alk (which type of alkali, K or Rb), T (the temp to calculate for), mol (1 for monomer, 2 for dimer)
	# Returns: The alkali density of a pure sample of "alk" molecules at a given temperature
def Nes_pure(alk, T, mol):
	CRC = (101325/1.38064852)*(10**17)
	if (mol == 1):
		if (alk == "K"):
			A = 13.83624
			B = 4857.902
			C = 0.0003494
			D = -2.21542
		elif (alk == "Rb"):
			A = 15.88253
			B = 4529.635
			C = 0.00058663
			D = -2.99138
		else:
			A, B, C, D = (0 for i in range(4))
			sys.exit("\n\n-----Invalid alkali type chosen; Nes_pure FAILS-----\n\n")
	elif (mol == 2):
		if (alk == "K"):
			A = 17.05231
			B = 6806.144
			C = 0.00012351
			D = -2.98966
		elif (alk == "Rb"):
			A = 41.2753
			B = 7226.316
			C = 0.00333213
			D = -11.8551
		else:
			A, B, C, D = (0 for i in range(4))
			sys.exit("\n\n-----Invalid alkali type chosen; Nes_pure FAILS-----\n\n")
	else:
		A, B, C, D = (0 for i in range(4))
		sys.exit("\n\n-----Invalid molecule type chosen; Nes_pure FAILS-----\n\n")
	return ( (CRC/T) * (10**( A - (B/T) + (C*T) + D*np.log10(T) ) ) )

# Notes "Dimer to Monomer Ratio"
	# Purpose: Gives ratio of dimers monomers in a pure, alkali vapor at a given temperature
	# Takes: alk (which type of alkali, K or Rb), T (the temp to calculate for)
	# Returns: Dimer to monomer ratio at temp T
def Nes_dim2mon(alk, T):
	T0 = 500
	if (alk == "K"):
		A2A1_0 = 0.0013
		b = 17.484
		c = 9.816
		d = -19.449
	elif (alk == "Rb"):
		A2A1_0 = 0.00282
		b = 11.862
		c = 2.482
		d = -7.623
	else:
		A2A1_0, b, c, d = (0 for i in range(4))
		sys.exit("\n\n-----Invalid alkali type chosen; Nes_dim2mon FAILS-----\n\n")
	return (A2A1_0 * np.exp( (-1.0*b*((T0/T) - 1)) + (c*((T/T0) - 1)) + (d*np.log10(T/T0)) ))

# Notes "Ratio of Pure Alkali Using CRC"
	# Purpose: Gives estimated ratio of pure alkali
	# Takes: temp (the temperature the data was taken at)
	# Returns: Ratio of pure K:Rb at given temp
def theoretical_CRC_Ratio(temp):
	T = float(temp) + 273.18
	return ( 10**(0.09 - (413 / T)) )

# Notes "Ratio of Pure Alkali Using Nesmayanov"
	# Purpose: Gives estimated ratio of pure alkali
	# Takes: temp (the temperature the data was taken at)
	# Returns: Ratio of pure K:Rb at given temp
def theoretical_Nes_Ratio(temp):
	T_data = float(temp) + 273.18
	Rb_Tdata_pure = (1.0/(Nes_dim2mon("Rb", T_data) + 1.0)) * Nes_pure("Rb", T_data, 1) + ( Nes_dim2mon("Rb", T_data) /(Nes_dim2mon("Rb", T_data) + 1.0)) * Nes_pure("Rb", T_data, 2)
	K_Tdata_pure = (1.0/(Nes_dim2mon("K", T_data) + 1.0)) * Nes_pure("K", T_data, 1) + ( Nes_dim2mon("K", T_data) /(Nes_dim2mon("K", T_data) + 1.0)) * Nes_pure("K", T_data, 2)
	return K_Tdata_pure/Rb_Tdata_pure

# Notes "Extrapolate to Operating Temperature"
	# Purpose: Extrapolates an alkali ratio to operating temperature using the CRC method
	# Takes: temp (the temperature the data was taken at), alkDenRat (the ratio of K:Rb from the data)
	# Returns: Ratio of K:Rb at operating temp (opTemp)
def extrapolateRatio_CRC(temp, alkDenRat):
	opTemp = 235 + 273.18
	temp = float(temp) + 273.18
	return (alkDenRat*10**(413.0*( (1/temp) - (1/opTemp) )))

# Notes "Extrapolate to Operating Temperature"
	# Purpose: Extrapolates an alkali ratio to operating temperature using the Nesmayanov method
	# Takes: temp (the temperature the data was taken at), alkDenRat (the ratio of K:Rb from the data)
	# Returns: Ratio of K:Rb at operating temp (opTemp)
def extrapolateRatio_Nesmayanov(temp, alkDenRat):
	data_pureRatio = theoretical_Nes_Ratio(temp)
	opTemp_pureRatio = theoretical_Nes_Ratio(235)
	return ( alkDenRat * (opTemp_pureRatio/data_pureRatio) )

##################

# Notes "Calculating R^2 for a Fit"
	# Purpose: Find R^2
	# Takes: y (the data), res (the residuals from the fit)
	# Returns: R^2 
def calcR2(y, res):
	ss_res = np.sum((res)**2)
	ss_tot = np.sum((y - np.mean(y))**2)
	return 1 - (ss_res/ss_tot)

# Notes "Helium-3 Density"
	# Purpose: Calculates the helium-3 density for a given spectral line
	# Takes: line (which line we're looking at), FWHM (the FWHM of the line), temp (temperature), N_den (N2 density from fill data), conSet (the set of constants to use)
	# Returns: Density of helium-3 calculated from this line
		# conSet options:	JD1, JD2, VN1, VN2, VN3, TA1, TA1_den (for the line shift, not FWHM)
		# line options:		RbD1, RbD2, KD1, KD2
def heliumDensity(line, FWHM, temp, N_den, conSet):
	if (conSet == "TA1_den"):
		if (line == "RbD1"):
			He_A = 5.46
			He_n = 0.38
			He_B = 0.24
			N_A = -7.65
			N_n = 0.44
			N_B = 0.25
		elif (line == "RbD2"):
			He_A = 0.63
			He_n = 1.42
			He_B = 0.20
			N_A = -5.70
			N_n = 0.48
			N_B = 0.23
		elif (line == "KD1"):
			He_A = 1.36
			He_n = -0.14
			He_B = 0.28
			N_A = -6.03
			N_n = 1.26
			N_B = 0.12
		elif (line == "KD2"):
			He_A = 0.69
			He_n = -2.04
			He_B = 0.02
			N_A = -5.04
			N_n = 0.72
			N_B = 0.05
		else:
			He_A = 1.0
			He_n = 0.0
			He_B = 0.0
			N_A = -1.0
			N_n = 0.0
			N_B = 0.0
	else:
		if (line == "RbD1"):
			if (conSet == "JD1"):
				He_A = 18.7
				He_n = 0.05
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "JD2"):
				He_A = 18.7
				He_n = 0.11
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN1"):
				He_A = 18.7
				He_n = 0.114
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN2"):
				He_A = 18.7
				He_n = 0.114
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN3"):
				He_A = 18.7
				He_n = 0.1668
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "TA1"):
				He_A = 18.31
				He_n = 0.26
				He_B = -0.19
				N_A = 17.41
				N_n = 0.3
				N_B = -0.49
			elif (conSet == "CJJ"):
				He_A = 65.97820691
				He_n = -4.48130225
				He_B = 0.0
				N_A = 160.60090257
				N_n = -4.48135334
				N_B = 0.0
			else:
				He_A = 1.0
				He_n = 0.0
				He_B = 0.0
				N_A = -2.0
				N_n = 0.0
				N_B = 0.0
		elif (line == "RbD2"):
			if (conSet == "JD1"):
				He_A = 20.8
				He_n = 0.53
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "JD2"):
				He_A = 20.8
				He_n = 0.34
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN1"):
				He_A = 20.8
				He_n = 0.344
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN2"):
				He_A = 20.8
				He_n = 0.344
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN3"):
				He_A = 20.8
				He_n = 0.40838
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "TA1"):
				He_A = 20.51
				He_n = 0.39
				He_B = -0.35
				N_A = 18.83
				N_n = -0.19
				N_B = -2.35
			elif (conSet == "CJJ"):
				He_A = -11591.2502
				He_n = -4.07703600
				He_B = 0.0
				N_A = 11806.1671
				N_n = -4.07703433
				N_B = 0.0
			else:
				He_A = 1.0
				He_n = 0.0
				He_B = 0.0
				N_A = -2.0
				N_n = 0.0
				N_B = 0.0
		elif (line == "KD1"):
			if (conSet == "JD1"):
				He_A = 14.4
				He_n = 0.41
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "JD2"):
				He_A = 14.4
				He_n = 0.41
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN1"):
				He_A = 14.381
				He_n = 0.4097
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN2"):
				He_A = 14.252
				He_n = 0.49649
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN3"):
				He_A = 14.484
				He_n = 0.43345
				He_B = 0.0
				N_A = 17.8
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "TA1"):
				He_A = 14.26
				He_n = 0.44
				He_B = 0.04
				N_A = 18.3
				N_n = 0.59
				N_B = -0.32
			elif (conSet == "CJJ"):
				He_A = 12024.5895
				He_n = -3.96376944
				He_B = 0.0
				N_A = -11875.1114
				N_n = -3.96377247
				N_B = 0.0
			else:
				He_A = 1.0
				He_n = 0.0
				He_B = 0.0
				N_A = -2.0
				N_n = 0.0
				N_B = 0.0
		elif (line == "KD2"):
			if (conSet == "JD1"):
				He_A = 20.15
				He_n = 0.23
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "JD2"):
				He_A = 20.15
				He_n = 0.23
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN1"):
				He_A = 20.154
				He_n = 0.231
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN2"):
				He_A = 19.523
				He_n = 0.5116
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "VN3"):
				He_A = 19.73
				He_n = 0.48539
				He_B = 0.0
				N_A = 18.1
				N_n = 0.3
				N_B = 0.0
			elif (conSet == "TA1"):
				He_A = 19.59
				He_n = 0.39
				He_B = 0.11
				N_A = 17.43
				N_n = 0.35
				N_B = 0.31
			elif (conSet == "CJJ"):
				He_A = -427.22516661
				He_n = -1.95130749
				He_B = 0.0
				N_A = 570.32468674   
				N_n = -1.9513839
				N_B = 0.0
			else:
				He_A = 1.0
				He_n = 0.0
				He_B = 0.0
				N_A = -2.0
				N_n = 0.0
				N_B = 0.0
		else:
			He_A = 1.0
			He_n = 0.0
			He_B = 0.0
			N_A = -1.0
			N_n = 0.0
			N_B = 0.0
	t = u"\u00b0"
	T = (float(temp) + 273.18)/353.0

	# Convert N2 Density to amagats
	N_den = (float(N_den)*0.00131579*273.15)/(1*293.15)
	
	he_den = ((FWHM - He_B - N_B - (float(N_den)*N_A*(T**N_n)))/(He_A*(T**He_n)))
	if (N_A == -1.0):
		print("INVALID LINE SELECTION: " + line)
	elif (N_A == -2.0):
		print("INVALID Constant Set: " + conSet)
	return he_den

# Notes "Plot Our Data"
	# Purpose: Plot data
	# Takes: line (which line are we plotting), xData, yData, yFit (line fitted to yData), residuals (between y and yFit)
	# Returns: None
def generatePlot(line, xData, yData, xFit, yFit, residuals, note):

	# Create plot of just this line
	plt.figure(mainFigureTitle + "_" + line)
	gs = gridspec.GridSpec(2,1, height_ratios=[4,1])

	ax0 = plt.subplot(gs[0])
	ax0.plot( xData, yData, color='r', marker='.', ms='.75', ls='', label=(line + "Data") )
	ax0.plot( xFit, yFit, color='k', marker=',', ls='-', lw='.5', label=(line + " Fit") )
	ax0.set_ylabel("LN(Primary:Reference)")

	# Annotate Main Plot
	xpos = plt.xlim()[0] + 0.05*(plt.xlim()[1] - plt.xlim()[0])
	ypos = plt.ylim()[0] + 0.05*(plt.ylim()[1] - plt.ylim()[0])
	ax0.annotate(note, (xpos, ypos))

	# Add residuals
	ax1 = plt.subplot(gs[1])
	ax1.plot( xData, residuals, color='b', marker=',', ls='' )
	ax1.set_ylabel("Residual")
	
	plt.xlabel( "Frequency (GHz)" )
	if (line == "RbD1"):
		lineStatement = " Rb D1 line at "
	elif (line == "RbD2"):
		lineStatement = " Rb D2 line at "
	else:
		lineStatement = " K lines at "
	plt.suptitle(cellName + lineStatement + temp + t + "C")

	if saveFigures:
		plt.show()
		plt.gcf().set_size_inches(15,7.5)
		plt.subplots_adjust(left=0.06,
		                    bottom=0.08, 
		                    right=0.98, 
		                    top=0.94, 
		                    wspace=0, 
		                    hspace=0)
		a, b = stats.chisquare(yData, f_exp=yFit)
		idx = ["RbD1", "RbD2", "KD1D2"].index(line)
		median = [0.06642581248416238, 0.13989421023239867, 3.444860826923521][idx]

		path = savePath
		chi2 = abs(a)
		if (chi2 == m.nan):
			path /= str("_nan")
		elif (chi2 < (median)):
			path /= str("_1median")
		elif (chi2 < (2*median)):
			path /= str("_2median")
		elif (chi2 < (3*median)):
			path /= str("_3median")
		elif (chi2 < (4*median)):
			path /= str("_4median")
		elif (chi2 < (5*median)):
			path /= str("_5median")
		elif (chi2 < (6*median)):
			path /= str("_6median")
		elif (chi2 < (7*median)):
			path /= str("_7median")
		elif (chi2 < (8*median)):
			path /= str("_8median")
		elif (chi2 < (9*median)):
			path /= str("_9median")
		else:
			path /= str("_lots")
		try:
			os.makedirs(path)
		except:
			pass
		os.chdir(path)
		plt.savefig(mainFigureTitle + "_" + line + "_" + fitUsed + ".png", dpi=600)
		plt.close()
		os.chdir(homePath)

# Notes "Integrate under the curve"
	# Purpose: Find the integral under the spectroscopy line, adjusted so it's relatively linear across the top
	# Takes: start (an index), stop (another index), wlm (a list of freq), ratio (a list of prim:ref ratios)
	# Returns: [integral, x value for peak, y value for peak, FWHM] 
			# Integral (the sum of the area under the natural log of hte adjusted spectroscopy line)
			# Where the peak (center) of the line is
			# What the value at the center is (depth of the line)
			# FWHM of the line
def calcIntegral(x, y):
	y1 = np.mean( y[0 : int(np.floor(0.1*len(y)))] )
	y2 = np.mean( y[int(np.floor(0.9*len(y))) : len(y)-1] )
	x1 = np.mean( x[0 : int(np.floor(0.1*len(x)))] )
	x2 = np.mean( x[int(np.floor(0.9*len(x))) : len(x)-1] )

	m = (y2 - y1) / (x2 - x1)
	b = 0.5*(y2+y1) - 0.5*m*(x2+x1)

	peak_x = 0
	peak_y = 99
	integral = 0
	for idx, val in enumerate(y):
		current_y = val - (m*x[idx] + b)
		if (current_y < peak_y):
			peak_x = x[idx]
			peak_y = current_y
		integral += current_y

	FWHM = 0
	
	for idx, val in enumerate(y):
		current_y = val - (m*x[idx] + b)
		if (current_y < (peak_y/2.0)):
			FWHM += 1
	return [integral, peak_x, peak_y, FWHM]

def stamp():
	startDate = re.sub('-', '', str(datetime.now().date()))
	startTime = str(datetime.now().time()).split(".")[0]
	startTime = startTime.split(":")[0] + "h" + startTime.split(":")[1] + "m" + startTime.split(":")[2] + "s"
	return f"D{startDate}_T{startTime}"

def logTheError(warning, line):
	a = ""
	fitNum = fitSelection
	global errorFlag
	if not errorFlag:
		a += f"File: {file}\t({fitNum}) {fitUsed}\n"
		errorFlag = True
	if len(sys.argv) > 4:
		errorFile = Path(savePath, "multiErrorLog.txt")
	else:
		errorFile = Path(homePath, "errorLog.txt")
	
	if warning == "opt":
		a += f"\t{stamp()}\t{line}\t---OPTIMIZATION FAILED\n"
	else:
		a += f"\t{stamp()}\t{line}\t---FIT FAILED\n"
	errorLog = open(errorFile, "a")
	errorLog.write(a)
	errorLog.close()

def monitor(note):
	print("\n" + note)
	return 0
'''----------------------------------------Main----------------------------------------
'''

############ Gather the Data
file = sys.argv[1]
data = np.loadtxt( file )

file = file.split("\\")[len(file.split("\\")) - 1]
file = file.split("/")[len(file.split("/")) - 1]
file = file.split(".")[0]
cellName = file.split("_")[0]
runNum = file.split("_")[1]
temp = re.sub('C', '', file.split("_")[2])

if updates:
	print()
	print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
	print()
	print("\n------------------  Beginning Analysis for File: " + file + "  ------------------\n")

# Start the main plots
mainFigureTitle = file
plt.figure(mainFigureTitle)
plt.suptitle( cellName + " at " + temp + t + "C")
plt.xlabel( "Frequency (GHz)" )
plt.ylabel( "LN(Primary:Reference)" )

### Split data into signal, wlm, errors, etc
ref = data[ :,1 ]
pri = data[ :,2 ]
wlm = data[ :,3 ]
ratio = np.log(pri/ref)
primToRef = pri/ref

### Initialize things that actually need to be initialized
wlm_RbD1, wlm_RbD2, wlm_KD1D2, ratio_RbD1, ratio_RbD2, ratio_KD1D2, primToRef_RbD1, primToRef_RbD2, primToRef_KD1D2 = ([] for _ in range(9))
ptsEx_RbD1, ptsEx_RbD2, ptsEx_KD1D2 = (0 for _ in range(3))

### Sort by line, remove bad data points (large error)
error = np.abs(data[ :,5 ])
for idx, val in enumerate(wlm):
	if (val<377808):
		if (error[idx] < allowedError):
			included_RbD1 = 1
			wlm_RbD1.append(val)
			ratio_RbD1.append(ratio[idx])
			primToRef_RbD1.append(primToRef[idx])
		else:
			ptsEx_RbD1 += 1
	elif (val<385243):
		if (error[idx] < allowedError):
			included_RbD2 = 1
			wlm_RbD2.append(val)
			ratio_RbD2.append(ratio[idx])
			primToRef_RbD2.append(primToRef[idx])
		else:
			ptsEx_RbD2 += 1
	else:
		if (error[idx] < allowedError):
			included_KD1 = 1
			if (val>391216):
				included_KD2 = 1
			wlm_KD1D2.append(val)
			ratio_KD1D2.append(ratio[idx])
			primToRef_KD1D2.append(primToRef[idx])
		else:
			ptsEx_KD1D2 += 1

del ratio
del primToRef
del error
del data

### Begin plotting any diagnostic plots
if plot_diagnostic:
	plt.figure(mainFigureTitle + "_primTOref")
	plt.suptitle( cellName + " at " + temp + t + "C")
	plt.xlabel( "Frequency (GHz)" )
	plt.ylabel( "Primary:Reference" )
	plt.figure(mainFigureTitle + "_Reference")
	plt.plot( wlm, ref, color='b', marker=',', ls='' )
	plt.figure(mainFigureTitle + "_Primary")
	plt.plot( wlm, pri, color='b', marker=',', ls='' )

	plt.figure(mainFigureTitle + "_primTOref")
	if included_RbD1:
		plt.plot( wlm_RbD1, primToRef_RbD1, color='b', marker=',', ls='' )
	if included_RbD2:
		plt.plot( wlm_RbD2, primToRef_RbD2, color='b', marker=',', ls='' )
	if included_KD1 or included_KD2:
		plt.plot( wlm_KD1D2, primToRef_KD1D2, color='b', marker=',', ls='' )

del wlm
del ref
del pri

### Cut a percentage from the wings (trust me, this is as efficient as it's going to be)
if (wingPercentCut > 0):
	cutSize_RbD1 = m.floor(wingPercentCut * len(wlm_RbD1) / 2)
	cutSize_RbD2 = m.floor(wingPercentCut * len(wlm_RbD2) / 2)
	cutSize_KD1D2 = m.floor(wingPercentCut * len(wlm_KD1D2) / 2)

	wlm_RbD1 = wlm_RbD1[cutSize_RbD1:(len(wlm_RbD1)-cutSize_RbD1)]
	wlm_RbD2 = wlm_RbD2[cutSize_RbD2:(len(wlm_RbD2)-cutSize_RbD2)]
	wlm_KD1D2 = wlm_KD1D2[cutSize_KD1D2:(len(wlm_KD1D2)-cutSize_KD1D2)]
	ratio_RbD1 = ratio_RbD1[cutSize_RbD1:(len(ratio_RbD1)-cutSize_RbD1)]
	ratio_RbD2 = ratio_RbD2[cutSize_RbD2:(len(ratio_RbD2)-cutSize_RbD2)]
	ratio_KD1D2 = ratio_KD1D2[cutSize_KD1D2:(len(ratio_KD1D2)-cutSize_KD1D2)]
	primToRef_RbD1 = primToRef_RbD1[cutSize_RbD1:(len(primToRef_RbD1)-cutSize_RbD1)]
	primToRef_RbD2 = primToRef_RbD2[cutSize_RbD2:(len(primToRef_RbD2)-cutSize_RbD2)]
	primToRef_KD1D2 = primToRef_KD1D2[cutSize_KD1D2:(len(primToRef_KD1D2)-cutSize_KD1D2)]

if updates:
	print("\n-----Number of Excluded Points-----\n")
	print("RbD1 Line:\t" + str(ptsEx_RbD1) + " of 1256 or " + str( np.round( 100.0*ptsEx_RbD1/1256, 3 )) + "%" )
	print("RbD2 Line:\t" + str(ptsEx_RbD2) + " of 2024 or " + str( np.round( 100.0*ptsEx_RbD2/2024, 3 )) + "%" )
	print("KD1D2 Line:\t" + str(ptsEx_KD1D2) + " of 3702 or " + str( np.round( 100.0*ptsEx_KD1D2/3702, 3 )) + "%\n")

# Line core for each spec line (averaged between isotopes for Rb)
lineCore_RbD1 = ( 0.2783 * 377107.463380 ) + ( 0.7217 * 377107.385690 ) # Rb-87 + Rb-85, weighted by natural abundance
lineCore_RbD2 = ( 0.2783 * 384230.484468 ) + ( 0.7217 * 384230.406373 ) # Rb-87 + Rb-85, weighted by natural abundance
lineCore_KD1 = ( 0.932581 *  389286.058716 ) + ( 0.000117 * 389286.184353 ) + ( 0.067302 * 389286.294205 ) # K-39 + K-40 + K-41, weighted by natural abundance
lineCore_KD2 = ( 0.932581 *  391016.17003 ) + ( 0.000117 * 391016.296050 ) + ( 0.067302 * 391016.40621 ) # K-39 + K-40 + K-41, weighted by natural abundance


#################################### the meat and potatoes

nullDict = {"best_vals":m.nan, "errs":m.nan, "yFit":m.nan, "resid":m.nan, "peak_x":m.nan, "peak_y":m.nan, "c0":m.nan, "FWHM":m.nan, "eta":m.nan, "fitType":"", "chi2":m.nan, "integral":m.nan}

if fitSelection ==1:
	print()
	print("\nSKIPPING Lorentzian_simple\n".center(400, "\'"))
	print()
	included_RbD1, included_RbD2, included_KD1, included_KD2, includeHeliumDensity, includeAlkaliDensityRatio = (False for _ in range(6))

### RbD1 ###
if(included_RbD1):
	x = np.asarray(wlm_RbD1) ### CHANGE ME IF COPY/PASTE
	y = np.asarray(ratio_RbD1) ### CHANGE ME IF COPY/PASTE
	
	if fitSelection > 0:
		#### CHANGE ME IF COPY/PASTE (next line down)						   0          1       2      3      4          5     6
		fitResults = fitterGetter(x, y, "RbD1", fitSelection) ### Returns: [yFit, best_vals, errors, D1Att, D2Att, residuals, flag] 
		if fitResults[6]:
			if ( "pV" in fitUsed.split("_") ):
				FWHM = (0.5346*fitResults[1][3] + np.sqrt(0.2166*fitResults[1][3]*fitResults[1][3] + fitResults[1][4]*fitResults[1][4])) ### 0.5346*fl + np.sqrt(0.2166*fl*fl + fg*fg)
				if (fitSelection == 5):
					f = ( (fitResults[1][4]**5) + 2.69269*(fitResults[1][4]**4)*(fitResults[1][3]) + 2.42843*(fitResults[1][4]**3)*(fitResults[1][3]**2) + 4.47163*(fitResults[1][4]**2)*(fitResults[1][3]**3) + 0.07842*(fitResults[1][4])*(fitResults[1][3]**4) + (fitResults[1][3]**5) )**(0.2)
					eta = ((1.36603*(fitResults[1][3]/f) - 0.47719*((fitResults[1][3]/f)**2) + 0.11116*((fitResults[1][3]/f)**3)))
				else:
					eta = (fitResults[1][5])
			else:
				FWHM = (fitResults[1][3])
				eta = 0
			chi2, pVal = stats.chisquare(y, f_exp=fitResults[0])
			d = {"best_vals":fitResults[1], "errs":fitResults[2], "yFit":fitResults[0], "resid":fitResults[5], "peak_x":fitResults[1][2], "peak_y":fitResults[3], "c0":fitResults[1][0], "FWHM":FWHM, "eta":eta, "fitType":fitUsed, "chi2":abs(chi2), "integral":0}

			# Add line to the big plots
			plt.figure(mainFigureTitle)
			plt.plot( x, y, color='r', marker='.', ms='.75', ls='', label=("Data") )
			plt.plot( x, fitResults[0], color='k', marker=',', ls='-', lw='.5', label=("Fit") )
		else:
			included_RbD1 = False
			d = copy.deepcopy(nullDict)
	else:
		fitResults = calcIntegral(x, y) ### returns: [integral, x value for peak, y value for peak, FWHM]
		d = {"integral":fitResults[0], "peak_x":fitResults[1], "peak_y":fitResults[2], "FWHM":fitResults[3], "chi2":0, "c0":0}

	### Integral Keys: ["integral", "peak_x", "peak_y", "FWHM"]
	### Everything Else Keys: "best_vals", "errs", "yFit", "residuals", "peak_x", "peak_y", "c0", "FWHM", "eta", "fitType"
	results_RbD1 = copy.deepcopy(d) #### CHANGE ME IF COPY/PASTE
	del d
else:
	results_RbD1 = copy.deepcopy(nullDict)

### RbD2 ###
if(included_RbD2):
	x = np.asarray(wlm_RbD2) ### CHANGE ME IF COPY/PASTE
	y = np.asarray(ratio_RbD2) ### CHANGE ME IF COPY/PASTE
	if fitSelection > 0:
		#### CHANGE ME IF COPY/PASTE (next line down)						   0          1       2      3      4          5     6
		fitResults = fitterGetter(x, y, "RbD2", fitSelection) ### Returns: [yFit, best_vals, errors, D1Att, D2Att, residuals, flag] 
		if fitResults[6]:
			if ( "pV" in fitUsed.split("_") ):
				FWHM = (0.5346*fitResults[1][3] + np.sqrt(0.2166*fitResults[1][3]*fitResults[1][3] + fitResults[1][4]*fitResults[1][4])) ### 0.5346*fl + np.sqrt(0.2166*fl*fl + fg*fg)
				if (fitSelection == 5):
					f = ( (fitResults[1][4]**5) + 2.69269*(fitResults[1][4]**4)*(fitResults[1][3]) + 2.42843*(fitResults[1][4]**3)*(fitResults[1][3]**2) + 4.47163*(fitResults[1][4]**2)*(fitResults[1][3]**3) + 0.07842*(fitResults[1][4])*(fitResults[1][3]**4) + (fitResults[1][3]**5) )**(0.2)
					eta = ((1.36603*(fitResults[1][3]/f) - 0.47719*((fitResults[1][3]/f)**2) + 0.11116*((fitResults[1][3]/f)**3)))
				else:
					eta = (fitResults[1][5])
			else:
				FWHM = (fitResults[1][3])
				eta = 0
			chi2, pVal = stats.chisquare(y, f_exp=fitResults[0])
			d = {"best_vals":fitResults[1], "errs":fitResults[2], "yFit":fitResults[0], "resid":fitResults[5], "peak_x":fitResults[1][2], "peak_y":fitResults[4], "c0":fitResults[1][0], "FWHM":FWHM, "eta":eta, "fitType":fitUsed, "chi2":abs(chi2), "integral":0}

			# Add line to the big plots
			plt.figure(mainFigureTitle)
			plt.plot( x, y, color='r', marker='.', ms='.75', ls='', label=("Data") )
			plt.plot( x, fitResults[0], color='k', marker=',', ls='-', lw='.5', label=("Fit") )	
		else:
			included_RbD2 = False
			d = copy.deepcopy(nullDict)
	else:
		fitResults = calcIntegral(x, y) ### returns: [integral, x value for peak, y value for peak, FWHM]
		d = {"integral":fitResults[0], "peak_x":fitResults[1], "peak_y":fitResults[2], "FWHM":fitResults[3], "chi2":0, "c0":0}

	results_RbD2 = copy.deepcopy(d)
	del d
else:
	results_RbD2 = copy.deepcopy(nullDict)

### KD1D2 ##
if(included_KD1 or included_KD2):
	x = np.asarray(wlm_KD1D2)
	y = np.asarray(ratio_KD1D2)

	if fitSelection > 0:							###						   0          1       2      3      4          5     6
		fitResults = fitterGetter(x, y, "KD1D2", fitSelection) ### Returns: [yFit, best_vals, errors, D1Att, D2Att, residuals, flag]
		if fitResults[6]:
			if ( "pV" in fitUsed.split("_") ):
				FWHM_D1 = (0.5346*fitResults[1][3] + np.sqrt(0.2166*fitResults[1][3]*fitResults[1][3] + fitResults[1][4]*fitResults[1][4])) ### 0.5346*fl + np.sqrt(0.2166*fl*fl + fg*fg)
				FWHM_D2 = (0.5346*fitResults[1][9] + np.sqrt(0.2166*fitResults[1][9]*fitResults[1][9] + fitResults[1][10]*fitResults[1][10])) ### 0.5346*fl + np.sqrt(0.2166*fl*fl + fg*fg)
				if (fitSelection == 5):
					f = ( (fitResults[1][4]**5) + 2.69269*(fitResults[1][4]**4)*(fitResults[1][3]) + 2.42843*(fitResults[1][4]**3)*(fitResults[1][3]**2) + 4.47163*(fitResults[1][4]**2)*(fitResults[1][3]**3) + 0.07842*(fitResults[1][4])*(fitResults[1][3]**4) + (fitResults[1][3]**5) )**(0.2)
					eta_D1 = ((1.36603*(fitResults[1][3]/f) - 0.47719*((fitResults[1][3]/f)**2) + 0.11116*((fitResults[1][3]/f)**3)))
					f = ( (fitResults[1][10]**5) + 2.69269*(fitResults[1][10]**4)*(fitResults[1][9]) + 2.42843*(fitResults[1][10]**3)*(fitResults[1][9]**2) + 4.47163*(fitResults[1][10]**2)*(fitResults[1][9]**3) + 0.07842*(fitResults[1][10])*(fitResults[1][9]**4) + (fitResults[1][9]**5) )**(0.2)
					eta_D2 = ((1.36603*(fitResults[1][9]/f) - 0.47719*((fitResults[1][9]/f)**2) + 0.11116*((fitResults[1][9]/f)**3)))
				else:
					eta_D1 = (fitResults[1][5])
					eta_D2 = (fitResults[1][11])
			else:
				FWHM_D1 = fitResults[1][3]
				FWHM_D2 = fitResults[1][7]
				eta_D1 = 0
				eta_D2 = 0
			chi2, pVal = stats.chisquare(y, f_exp=fitResults[0])
			d1 = {"best_vals":fitResults[1], "errs":fitResults[2], "yFit":fitResults[0], "resid":fitResults[5], "peak_x":fitResults[1][2], "peak_y":fitResults[3], "c0":fitResults[1][0], "FWHM":FWHM_D1, "eta":eta_D1, "fitType":fitUsed, "chi2":abs(chi2), "integral":0}
			d2 = {"best_vals":fitResults[1], "errs":fitResults[2], "yFit":fitResults[0], "resid":fitResults[5], "peak_x":fitResults[1][6], "peak_y":fitResults[4], "c0":fitResults[1][4], "FWHM":FWHM_D2, "eta":eta_D2, "fitType":fitUsed, "chi2":abs(chi2), "integral":0}
			# Add line to the big plot
			plt.figure(mainFigureTitle)
			plt.plot( x, y, color='r', marker='.', ms='.75', ls='', label=("Data") )
			plt.plot( x, fitResults[0], color='k', marker=',', ls='-', lw='.5', label=("Fit") )	
		else:
			included_KD1, included_KD2 = (False for _ in range(2))
			d1 = copy.deepcopy(nullDict)
			d2 = copy.deepcopy(nullDict)
	else:
		halfIdx = 0
		for idx, val in enumerate(wlm_KD1D2):
			if (val>390283):
				halfIdx = idx
				break
		# Gather data for integrated solution
		fitResults = calcIntegral(x[:halfIdx], y[:halfIdx])
		d1 = {"integral":fitResults[0], "peak_x":fitResults[1], "peak_y":fitResults[2], "FWHM":fitResults[3], "chi2":0, "c0":0}
		fitResults = calcIntegral(x[halfIdx:], y[halfIdx:])
		d2 = {"integral":fitResults[0], "peak_x":fitResults[1], "peak_y":fitResults[2], "FWHM":fitResults[3], "chi2":0, "c0":0}

	### Integral Keys: ["integral", "peak_x", "peak_y", "FWHM"]
	### Everything Else Keys: "best_vals", "covar", "yFit", "residuals", "peak_x", "peak_y", "c0", "FWHM", "eta", "fitType", "R2", "X2", "pVal"
	results_KD1 = copy.deepcopy(d1)
	results_KD2 = copy.deepcopy(d2)
	del d1
	del d2
else:
	results_KD1 = copy.deepcopy(nullDict)
	results_KD2 = copy.deepcopy(nullDict)

''' ------------------- RESULTS ----------------------- '''

if updates:
	print("\n\n------------------------------------------Results for " + file + " ------------------------------------------\n\n")

s_He, s_Alk, sRD1, sRD2, sKD1, sKD2 = ('' for i in range(6))
RbD1_3HeDen, RbD2_3HeDen, KD1_3HeDen, KD2_3HeDen = (0 for _ in range(4))

copyPaste = "\n\n" + " COPY/PASTE ".center(80, "-") + "\n" + "-------- FWHM (RbD1, RbD2, KD1, KD2) --------" + "\n"
copyPaste += str(results_RbD1["FWHM"]) + "\n" + str(results_RbD2["FWHM"]) + "\n" + str(results_KD1["FWHM"]) + "\n" + str(results_KD2["FWHM"])

if updates:
	print("                                           Fit Used: " + fitUsed)

### Find 3He Density
if ( (len(sys.argv)>2) and includeHeliumDensity ):

	#### Add to Copy/Paste Section
	copyPaste += "\n\n-------- 3He Density (RbD1, RbD2, KD1, KD2) ----------------"
	allDensity = []
	N_den = float(sys.argv[2])*(273.15/293.15)/760  # N2 density in amagats
	s_He += "3He Density:"

	if(included_RbD1):
		if (conSet == "TA1_den"):
			### Use Freq Shift (Don't use this)
			FWHM = np.abs( lineCore_RbD1 - results_RbD1["peak_x"] )
		else:
			### Subtract Natural Line Width
			FWHM = (results_RbD1["FWHM"] - 0.0057500 )
		RbD1_3HeDen = heliumDensity("RbD1", FWHM, temp, N_den, conSet)
		allDensity.append(RbD1_3HeDen)
		sRD1 += "\n  Estimated 3He Density: " + str(np.round(RbD1_3HeDen, 3)) + " amgts"
		copyPaste += "\n" + str(RbD1_3HeDen)
	if(included_RbD2):
		if (conSet == "TA1_den"):
			### Use Freq Shift (Don't use this)
			FWHM = np.abs( lineCore_RbD2 - results_RbD2["peak_x"] )
		else:
			### Subtract Natural Line Width
			FWHM = (results_RbD2["FWHM"] - 0.0060666 )
		RbD2_3HeDen = heliumDensity("RbD2", FWHM, temp, N_den, conSet)
		allDensity.append(RbD2_3HeDen)
		sRD2 += "\n  Estimated 3He Density: " + str(np.round(RbD2_3HeDen, 3)) + " amgts"
		copyPaste += "\n" + str(RbD2_3HeDen)
	if(included_KD1):
		if (conSet == "TA1_den"):
			### Use Freq Shift (Don't use this)
			FWHM = np.abs( lineCore_KD1 - results_KD1["peak_x"] )
		else:
			### Subtract Natural Line Width
			FWHM = (results_KD1["FWHM"] - 0.005956 )
		KD1_3HeDen = heliumDensity("KD1", FWHM, temp, N_den, conSet)
		allDensity.append(KD1_3HeDen)
		sKD1 += "\n  D1 Estimated 3He Density: " + str(np.round(KD1_3HeDen, 3)) + " amgts"
		copyPaste += "\n" + str(KD1_3HeDen)
	if(included_KD2):
		if (conSet == "TA1_den"):
			### Use Freq Shift (Don't use this)
			FWHM = np.abs( lineCore_KD2 - results_KD2["peak_x"] )
		else:
			### Subtract Natural Line Width
			FWHM = (results_KD2["FWHM"] - 0.006035 )
		KD2_3HeDen = heliumDensity("KD2", FWHM, temp, N_den, conSet)
		allDensity.append(KD2_3HeDen)
		sKD2 += "\n  D2 Estimated 3He Density: " + str(np.round(KD2_3HeDen, 3)) + " amgts"
		copyPaste += "\n" + str(KD2_3HeDen)

	############# Fitted Results for 3He Density #############

	if len(allDensity)>1:
		s_He += "\n   Combined: " + str(np.round(np.mean(allDensity), 3)) + " +/- " + str(np.round(np.std(allDensity), 3)) + " amgts"

	#### Report Results
	if updates:
		print("---------------------3He Density Results (in amgts) and FWHM (in GHz)---------------------")

		if included_RbD1:
			print("\tRbD1: " + str(np.round(RbD1_3HeDen, 3)) + " (FWHM: " + str(np.round(results_RbD1["FWHM"], 3)) + ")")
		if included_RbD2:
			print("\tRbD2: " + str(np.round(RbD2_3HeDen, 3)) + " (FWHM: " + str(np.round(results_RbD2["FWHM"], 3)) + ")")
		if included_KD1:
			print("\tKD1: " + str(np.round(KD1_3HeDen, 3)) + " (FWHM: " + str(np.round(results_KD1["FWHM"], 3)) + ")")
		if included_KD2:
			print("\tKD2: " + str(np.round(KD2_3HeDen, 3)) + " (FWHM: " + str(np.round(results_KD2["FWHM"], 3)) + ")")
		print("\t\tAveraged Result: " + str(np.round(np.mean(allDensity), 3)) + " +/- " + str(np.round(np.std(allDensity), 3)) + " amgts" )

alkRatio_raw_mean, alkRatio_CRC_mean, alkRatio_nes_mean, alkRatio_raw_stdv, alkRatio_CRC_stdv, alkRatio_nes_stdv = (m.nan for _ in range(6))

### Find Alkali Density Ratio
if(includeAlkaliDensityRatio and (len(wlm_KD1D2) > 0)):

	alkRatio_raw, alkRatio_CRC, alkRatio_nes = ([] for i in range(3))
	T = float(temp)

	# Oscillator strength from "Optical Pumping (Happer)" pg 218
	f_RbD1 = 0.35
	f_RbD2 = 0.70
	f_KD1 = 0.34
	f_KD2 = 0.68

	flag = False

	if fitSelection == 0:
		if included_RbD1 and included_KD1:
			alk_ratio = (f_RbD1/f_KD1) * (results_KD1["integral"]/results_RbD1["integral"])
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD1 and included_KD2:
			alk_ratio = (f_RbD1/f_KD2) * (results_KD2["integral"]/results_RbD1["integral"])
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD2 and included_KD1:
			alk_ratio = (f_RbD2/f_KD1) * (results_KD1["integral"]/results_RbD2["integral"])
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD2 and included_KD2:
			alk_ratio = (f_RbD2/f_KD2) * (results_KD2["integral"]/results_RbD2["integral"])
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
	else:
		if included_RbD1 and included_KD1:
			alk_ratio = (f_RbD1/f_KD1) * ( (results_KD1["c0"]*results_RbD1["FWHM"])/(results_KD1["FWHM"]*results_RbD1["c0"]) ) 
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD1 and included_KD2:
			alk_ratio = (f_RbD1/f_KD2) * ( (results_KD2["c0"]*results_RbD1["FWHM"])/(results_KD2["FWHM"]*results_RbD1["c0"]) ) 
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD2 and included_KD1:
			alk_ratio = (f_RbD2/f_KD1) * ( (results_KD1["c0"]*results_RbD2["FWHM"])/(results_KD1["FWHM"]*results_RbD2["c0"]) ) 
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
		if included_RbD2 and included_KD2:
			alk_ratio = (f_RbD2/f_KD2) * ( (results_KD2["c0"]*results_RbD2["FWHM"])/(results_KD2["FWHM"]*results_RbD2["c0"]) ) 
			alkRatio_raw.append( alk_ratio )
			alkRatio_CRC.append( extrapolateRatio_CRC(T, alk_ratio) )
			alkRatio_nes.append( extrapolateRatio_Nesmayanov(T, alk_ratio) )
			flag = True
	if flag:
		alkRatio_raw_mean = np.mean(alkRatio_raw)
		alkRatio_CRC_mean = np.mean(alkRatio_CRC)
		alkRatio_nes_mean = np.mean(alkRatio_nes)
		alkRatio_raw_stdv = np.std(alkRatio_raw)
		alkRatio_CRC_stdv = np.std(alkRatio_CRC)
		alkRatio_nes_stdv = np.std(alkRatio_nes)

		#### Add to Annotation
		s_Alk = "Alkali Ratio (K:Rb):" 
		s_Alk += "\n   Fitted Raw: " + str(np.round(alkRatio_raw_mean, 3)) + " +/- " + str(np.round(alkRatio_raw_stdv, 3))
		s_Alk += "\n   Fitted Projected (CRC) (235" + t + "C): " + str(np.round(alkRatio_CRC_mean, 3)) + " +/- " + str(np.round(alkRatio_CRC_stdv, 3))
		s_Alk += "\n   Fitted Projected (Nes) (235" + t + "C): " + str(np.round(alkRatio_nes_mean, 3)) + " +/- " + str(np.round(alkRatio_nes_stdv, 3))

		#### Report Results
		if updates:
			print("\n\n---------------------Alkali Density Ratio (K:Rb)---------------------")
			print("\tFitted Raw: " + str(np.round(alkRatio_raw_mean, 3)) + " +/- " + str(np.round(alkRatio_raw_stdv, 3)))
			print("\tFitted Projected (CRC) (235" + t + "C): " + str(np.round(alkRatio_CRC_mean, 3)) + " +/- " + str(np.round(alkRatio_CRC_stdv, 3)))
			print("\tFitted Projected (Nes) (235" + t + "C): " + str(np.round(alkRatio_nes_mean, 3)) + " +/- " + str(np.round(alkRatio_nes_stdv, 3)))

		#### Add to Copy/Paste Section
		copyPaste += "\n\n-------- Alkali Density Ratio - [raw, CRC, nes, raw_std, CRC_std, nes_std]----------------"
		copyPaste += "\n" + str(alkRatio_raw_mean) + "\n" + str(alkRatio_CRC_mean) + "\n" + str(alkRatio_nes_mean) + "\n" + str(alkRatio_raw_stdv) + "\n" + str(alkRatio_CRC_stdv) + "\n" + str(alkRatio_nes_stdv)

if updates:
	print(copyPaste)
	print("\n\n")
	print("---------------------------- " + mainFigureTitle + " (" + str(fitSelection) + ") " + fitUsed + " Complete ----------------------------")

exportDict = {"RbD1_FWHM":results_RbD1["FWHM"], "RbD1_center":results_RbD1["peak_x"], "RbD1_c0":results_RbD1["c0"], "RbD1_integral":results_RbD1["integral"], "RbD1_chi2":results_RbD1["chi2"], "RbD2_FWHM":results_RbD2["FWHM"], "RbD2_center":results_RbD2["peak_x"], "RbD2_c0":results_RbD2["c0"], "RbD2_integral":results_RbD2["integral"], "RbD2_chi2":results_RbD2["chi2"], "KD1_FWHM":results_KD1["FWHM"], "KD1_center":results_KD1["peak_x"], "KD1_c0":results_KD1["c0"], "KD1_integral":results_KD1["integral"], "KD1_chi2":results_KD1["chi2"], "KD2_FWHM":results_KD2["FWHM"], "KD2_center":results_KD2["peak_x"], "KD2_c0":results_KD2["c0"], "KD2_integral":results_KD2["integral"], "KD2_chi2":results_KD2["chi2"], "T":temp, "Target":cellName, "runNum":runNum}
export_toPickle(exportDict, "")

if updates:
	print("\n\n--- Values in exportDict ----\n")
	for key, val in exportDict.items():
		print(f"\t{key}\t:\t{val}")
	print("\n--- End of Values ---\n\n")

### Integral Keys: ["integral", "peak_x", "peak_y", "FWHM"]
### Everything Else Keys: "best_vals", "covar", "yFit", "residuals", "peak_x", "peak_y", "c0", "FWHM", "eta", "fitType", "R2", "X2", "pVal"



### Select plots to show
if(includePlots and (fitSelection > 0) and (included_RbD1 or included_RbD2 or included_KD1 or included_KD2) ):
	monitor(f"\n\nPlotting")
	### Generate plots for individual lines
	if plot_eachLine:
		s = ""
		if included_RbD1:
			if annotated: s = "Results: \n  FWHM: " + str(np.round(results_RbD1["FWHM"], 3)) + " GHz" + sRD1
			generatePlot("RbD1", np.asarray(wlm_RbD1), np.asarray(ratio_RbD1), np.asarray(wlm_RbD1), results_RbD1["yFit"], results_RbD1["resid"], s)
		if included_RbD2:
			if annotated: s = "Results: \n  FWHM: " + str(np.round(results_RbD2["FWHM"], 3)) + " GHz" + sRD2
			generatePlot("RbD2", np.asarray(wlm_RbD2), np.asarray(ratio_RbD2), np.asarray(wlm_RbD2), results_RbD2["yFit"], results_RbD2["resid"], s)
		if included_KD1 or included_KD2:
			if annotated: s = "Results: \n  D1 FWHM: " + str(np.round(results_KD1["FWHM"], 3)) + " GHz" + sKD1 + "\n  D2 FWHM: " + str(np.round(results_KD2["FWHM"], 3)) + " GHz" + sKD2
			generatePlot("KD1D2", np.asarray(wlm_KD1D2), np.asarray(ratio_KD1D2), np.asarray(wlm_KD1D2), results_KD1["yFit"], results_KD1["resid"], s)
		### Annotate Plots
	if annotated:
		plt.figure(mainFigureTitle)
		s = ""
		if includeHeliumDensity:
			s += s_He + "\n"
		if includeAlkaliDensityRatio:
			s += s_Alk + "\n"
		s += "FWHM (GHz):"
		if included_RbD1:
			s += "\n   RbD1: " + str(np.round(results_RbD1["FWHM"], 3))
			s += "\n     chi2: " + str(np.round(abs(results_RbD1["chi2"]), 4))
		if included_RbD2:
			s += "\n   RbD2: " + str(np.round(results_RbD2["FWHM"], 3))
			s += "\n     chi2: " + str(np.round(abs(results_RbD2["chi2"]), 4))
		if included_KD1:
			s += "\n   KD1: " + str(np.round(results_KD1["FWHM"], 3))
			s += "\n     chi2: " + str(np.round(abs(results_KD1["chi2"]), 4))
		if included_KD2:
			s += "\n   KD2: " + str(np.round(results_KD2["FWHM"], 3))
			s += "\n     chi2: " + str(np.round(abs(results_KD2["chi2"]), 4))

		xpos = plt.xlim()[0] + 0.1*(plt.xlim()[1] - plt.xlim()[0])
		ypos = plt.ylim()[0] + 0.1*(plt.ylim()[1] - plt.ylim()[0])
		plt.annotate(s, (xpos, ypos))

	plt.figure(mainFigureTitle)
	plt.gcf().canvas.manager.set_window_title(str(mainFigureTitle) + "_" + str(fitUsed))
	### Show figures (single run) or save the main figure (muti-run)
	plt.show()
	if saveFigures:
		plt.gcf().set_size_inches(15,7.5)
		plt.subplots_adjust(left=0.06,
		                    bottom=0.08, 
		                    right=0.98, 
		                    top=0.94, 
		                    wspace=0, 
		                    hspace=0)
		os.chdir(savePath)
		plt.savefig(mainFigureTitle + "_" + fitUsed + ".png", dpi=600)
		plt.close()
		os.chdir(homePath)
gc.collect()
if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()


