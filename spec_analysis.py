'''----------------------------------------Imports----------------------------------------
'''

import sys, re, os
import numpy as np
import pickle
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import copy
import random as rnum
t = u"\u00b0"
pm = " " + u"\u00B1" + " "

from sympy.physics.wigner import wigner_6j
from sympy.physics.wigner import wigner_3j

'''----------------------------------------Flags and Options----------------------------------------
'''
tryCount = 0
allowedError = 1
updates = False
attemptsToFit = 100000
plot_show = False
plot_save = True
plot_each = False
plot_diagnostic = False

homePath = Path.cwd()
analysisPath = Path(homePath, "__AllData", "__analysis")
plotPath = Path(analysisPath, "plots")
picklePath = Path(analysisPath, "data_pickles")
logPath = Path(analysisPath, "logs")
listOfPaths = [plotPath, homePath, picklePath, logPath]

init_c0 = 0 	### Loops in steps of -100 until abs(chi2) < chi2_goal
init_c1 = 0
in_FWHM = 0 ###	Set at line 268 	(Kappa = 16.0,	else = 119.0)
in_eta = 0.61
in_m = 0.00000133
in_b = -0.67969

in_step_c0 = -100.0
in_step_c1 = 2
gran_c0 = 5.0 
gran_c1 = 2.0

chi2_goal = 1
iterator_max = 20
acceptableDiff = 1
chi2_threshold = 0

FWHM_LB = 100
FWHM_HB = 0


'''----------------------------------------Functions----------------------------------------
'''

def import_unpickle(fileIn):
	with open(fileIn, 'rb') as x:
		temp = pickle.load(x)
	return temp

def export_pickle(n, file):
	with open(file, 'wb') as e:
		pickle.dump(n, e)

def monitor(t):
	for i, v in enumerate(t):
		print(f"\t--> {v}")

def pathChecker(pList):
	for i, p in enumerate(pList):
		try:
			os.makedirs(p)
		except:
			pass

#####################

def Rb_lor( x, c0, c1, c2, c3, m, b):
	del1 = x - c2 + 1.264887
	del2 = x - c2 - 1.770884
	del3 = x - c2 + 2.563005
	del4 = x - c2 - 4.271676
	part1 = (1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2)
	part2 = (1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2)
	part3 = (1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2)
	part4 = (1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2)
	return (0.7217*c0*( (7/12)*part1 + (5/12)*part2 ) + 0.2783*c0*( (5/8)*part3 + (3/8)*part4 ) + m*x + b )

def K_lor( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
	D1 = (1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)
	D2 = (1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)
	return (D1c0*D1 + D2c0*D2 + m*x + b)

def Rb_pV( x, c0, c1, c2, c3, eta, m, b):
	del1 = x - c2 + 1.264887
	del2 = x - c2 - 1.770884
	del3 = x - c2 + 2.563005
	del4 = x - c2 - 4.271676

	part1 = eta*( (1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del1**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del1)))**2) )
	part2 = eta*( (1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del2**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del2)))**2) )
	part3 = eta*( (1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del3**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del3)))**2) )
	part4 = eta*( (1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2) ) + (1-eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/c3)**2) * np.exp( (-((2*np.log(2))**2)*(del4**2)) / ((c3*(1 + (0.664*2*np.pi*c1*del4)))**2) )
	return (0.7217*c0*( (7/12)*part1 + (5/12)*part2 ) + 0.2783*c0*( (5/8)*part3 + (3/8)*part4 ) + m*x + b )

def K_pV( x, D1c0, D1c1, D1c2, D1c3, D1eta, D2c0, D2c1, D2c2, D2c3, D2eta, m, b):
	D1 = D1eta*((1 + 0.664*2*np.pi*D1c1*(x-D1c2)) / ((x-D1c2)**2 + (D1c3/2)**2)) + (1-D1eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/D1c3)**2) * np.exp( (-((2*np.log(2))**2)*((x-D1c2)**2)) / ((D1c3*(1 + (0.664*2*np.pi*D2c1*(x-D1c2))))**2) )
	D2 = D2eta*((1 + 0.664*2*np.pi*D2c1*(x-D2c2)) / ((x-D2c2)**2 + (D2c3/2)**2)) + (1-D2eta) * (np.sqrt(8*np.pi)*np.log(2)) * ((1/D2c3)**2) * np.exp( (-((2*np.log(2))**2)*((x-D2c2)**2)) / ((D2c3*(1 + (0.664*2*np.pi*D2c1*(x-D2c2))))**2) )
	return (D1c0*D1 + D2c0*D2 + m*x + b)

#####################

def Rb_lor_full( x, c0, c1, c2, c3, m, b, line):
	line = int(line)
	J = [0, 1/2, 3/2][line]
	
	Rb85, Rb87 = (0 for _ in range(2))

	### Rb85 ###
	I = 5/2
	temp_f = []
	temp_d = []
	temp = []
	low = int([0,2,1][line])
	high = int([0,4,5][line])
	for f_g in range(2,4):
		for f_e in range(low,high):
			temp_f.append(float(Rb_HFtranStrength(f_g, f_e, J, I)))
			insert = (x - c2 + float(Rb_tranShift(f_g, f_e, "85", line)))
			temp_d.append( (1 + (0.664*2*np.pi*c1*insert)) / (insert**2 + (c3/2)**2) )
	for idx, val in enumerate(temp_f):
		Rb85 += val*temp_d[idx]

	### Rb87 ###
	I = 3/2
	temp_f = []
	temp_d = []
	temp = []
	low = int([0,1,0][line])
	high = int([0,3,4][line])
	for f_g in range(1,3):
		for f_e in range(low,high):
			if ((f_e == f_g) or (abs(f_e-f_g) == 1)):
				temp_f.append(float(Rb_HFtranStrength(f_g, f_e, J, I)))
				insert = (x - c2 + float(Rb_tranShift(f_g, f_e, "87", line)))
				temp_d.append( (1 + (0.664*2*np.pi*c1*insert)) / (insert**2 + (c3/2)**2) )
	for idx, val in enumerate(temp_f):
		Rb87 += val*temp_d[idx]

	return ( 0.7217*c0*(Rb85) + 0.2783*c0*(Rb87) + m*x + b)

def Rb_HFtranStrength(f_g, f_e, j_e, I):
	return (2*f_e + 1)*(2*(1/2) + 1)*(wigner_6j(1/2, j_e, 1, f_e, f_g, I)**2)

def Rb_tranShift(f_g, f_e, isotope, line): ### all freq in GHz
	isoShift, g, e = ([] for _ in range(3))
	if isotope == "85":
		isoShift = [0, -0.0216211269726045, -0.0217339776572771] ## Shift from average peak for [nan,D1,D2] lines
		g = [0, 0, -1.7708439228, 1.2648885163] 
		e = [[0], [0, 0, -0.210923, 0.150659], [0, -0.113208, -0.083835, -0.20435, 0.100205]]
	elif isotope == "87":
		isoShift = [0, 0.0560688729747199, 0.0563615223509260]
		g = [0, -4.271676631815181, 2.563005979089109]
		e = [[0], [0, -0.50906, 0.30544], [-0.3020738, -0.2298518, -0.0729112, 0.1937407]]
	else:
		print("NOT A VALID ISOTOPE")
		return 0
	return (isoShift[int(line)] + e[int(line)][int(f_e)] + g[int(f_g)])

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
	return [integral, peak_y, peak_x, FWHM]

def exportPlots(showPlots, savePlots, everyLine):
	plt.figure(file)
	if everyLine:
		plt.close()
		for idx, line in enumerate(lines):
			if included[idx]:
				if idx == 2:
					s_param = "D1 FWHM: " + str(np.round(dictOfResultsDicts["KD1"]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["FWHM_std"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts["KD1"]["center"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["center_std"], 3)) + " GHz"
					if included[3]:
						s_param += "\n\nD2 FWHM: " + str(np.round(dictOfResultsDicts["KD2"]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["FWHM_std"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts["KD2"]["center"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["center_std"], 3)) + " GHz"
					s_stats = "chi2: " + str(np.round(dictOfResultsDicts["KD1"]["chi2"], 3)) + "\npVal: "  + str(np.round(dictOfResultsDicts["KD1"]["pVal"], 3))
					s = s_param + "\n\n" + s_stats
					generatePlot(line, dictOfResultsDicts["KD1"]["x"], dictOfResultsDicts["KD1"]["y"], dictOfResultsDicts["KD1"]["yFit"], s)
				else:
					s_param = "FWHM: " + str(np.round(dictOfResultsDicts[line]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts[line]["FWHM_std"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts[line]["center"], 3)) + pm + str(np.round(dictOfResultsDicts[line]["center_std"], 3)) + " GHz"
					s_stats = "chi2: " + str(np.round(dictOfResultsDicts[line]["chi2"], 3)) + "\npVal: "  + str(np.round(dictOfResultsDicts[line]["pVal"], 3))
					s = s_param + "\n\n" + s_stats
					generatePlot(line, dictOfResultsDicts[line]["x"], dictOfResultsDicts[line]["y"], dictOfResultsDicts[line]["yFit"], s)
	else:
		s_FWHM = "FWHM (GHz):"
		s_center = "Center:"
		s_stats = "Stats:"
		for idx, line in enumerate(lines):
			if included[idx]:
				if idx == 2:
					s_FWHM += f"\n   KD1: " + str(np.round(dictOfResultsDicts["KD1"]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["FWHM_std"], 3))
					s_center += f"\n   KD1: " + str(np.round(dictOfResultsDicts["KD1"]["center"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["center_std"], 3))
					if included[3]:
						s_FWHM += f"\n   KD2: " + str(np.round(dictOfResultsDicts["KD2"]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["FWHM_std"], 3))
						s_center += f"\n   KD2: " + str(np.round(dictOfResultsDicts["KD2"]["center"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["center_std"], 3))
					s_stats += f"\n   K-Lines: [chi2: " + str(np.round(abs(dictOfResultsDicts["KD1"]["chi2"]), 4)) + "]  [pVal: " + str(np.round(dictOfResultsDicts["KD1"]["pVal"], 4)) + "]"
				else:
					s_FWHM += f"\n   {line}: " + str(np.round(dictOfResultsDicts[line]["FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts[line]["FWHM_std"], 3))
					s_center += f"\n   {line}: " + str(np.round(dictOfResultsDicts[line]["center"], 3)) + pm + str(np.round(dictOfResultsDicts[line]["center_std"], 3))
					s_stats += f"\n   {line}: [chi2: " + str(np.round(abs(dictOfResultsDicts[line]["chi2"]), 4)) + "]  [pVal: " + str(np.round(dictOfResultsDicts[line]["pVal"], 4)) + "]"

		xpos = plt.xlim()[0] + 0.1*(plt.xlim()[1] - plt.xlim()[0])
		ypos = plt.ylim()[0] + 0.1*(plt.ylim()[1] - plt.ylim()[0])
		s = s_FWHM + "\n\n" + s_center + "\n\n" + s_stats
		plt.annotate(s, (xpos, ypos))

		plt.gcf().canvas.manager.set_window_title(str(file) + "_" + str(fitUsed))
		### Show figures (single run) or save the main figure (muti-run)

		if savePlots:
			plt.gcf().set_size_inches(15,7.5)
			plt.subplots_adjust(left=0.06,
			                    bottom=0.08, 
			                    right=0.98, 
			                    top=0.94, 
			                    wspace=0, 
			                    hspace=0)
			os.chdir(plotPath)
			plt.savefig(file + "_" + fitUsed + ".png", dpi=600)
			os.chdir(homePath)
		if showPlots:
			plt.show()
		plt.close()

def generatePlot(line, xData, yData, yFit, note):

	# Create plot of just this line
	plt.figure(file + "_" + line)
	gs = gridspec.GridSpec(2,1, height_ratios=[4,1])

	ax0 = plt.subplot(gs[0])
	ax0.plot( xData, yData, color='r', marker='.', ms='.75', ls='', label=(line + "Data") )
	ax0.plot( xData, yFit, color='k', marker=',', ls='-', lw='.5', label=(line + " Fit") )
	ax0.set_ylabel("LN(Primary:Reference)")

	# Annotate Main Plot
	xpos = plt.xlim()[0] + 0.05*(plt.xlim()[1] - plt.xlim()[0])
	ypos = plt.ylim()[0] + 0.05*(plt.ylim()[1] - plt.ylim()[0])
	ax0.annotate(note, (xpos, ypos))

	# Add residuals
	residuals = yFit-yData
	ax1 = plt.subplot(gs[1])
	ax1.plot( xData, residuals, color='b', marker=',', ls='' )
	ax1.set_ylabel("Residual")
	
	plt.xlabel( "Frequency (GHz)" )
	if (line == "RbD1"):
		subPlotPath = plotPath / "RbD1"
		lineStatement = " Rb D1 line at "
	elif (line == "RbD2"):
		subPlotPath = plotPath / "RbD2"
		lineStatement = " Rb D2 line at "
	else:
		subPlotPath = plotPath / "KD1D2"
		lineStatement = " K lines at "
	plt.suptitle(cellName + lineStatement + temp + t + "C")
	if plot_show:
		plt.show()
	if plot_save:
		plt.gcf().set_size_inches(15,7.5)
		plt.subplots_adjust(left=0.06,
		                    bottom=0.08, 
		                    right=0.98, 
		                    top=0.94, 
		                    wspace=0, 
		                    hspace=0)
		try:
			os.chdir(subPlotPath)
		except:
			os.makedirs(subPlotPath)
			os.chdir(subPlotPath)
		plt.savefig(file + "_" + line + "_" + fitUsed + ".png", dpi=600)
		os.chdir(homePath)
	plt.close()

def logger(d, dataSet, fit):
	global fitSelect
	s = f"---> {dataSet} - ({int(sys.argv[2])}) {fit} <---\n"
	for k, v in exportDict.items():
		if ("_x" in k) or ("_y" in k):# or ("_c0" in k) or ("_c1" in k) or ("_b" in k) or ("_integral" in k) or ("_chi2" in k) or ("_eta" in k):
			pass
		elif "maxAtt" in k:
			s += f"key: {k}\tval: {v}\n\n"
		else:
			s += f"key: {k}\tval: {v}\n"
	f = Path(logPath, "outputRecord.txt")
	s += "\n\n\n"
	writer = open(f, "a")
	writer.write(s)
	writer.close()
	if updates:
		print(s)

def alkDenRatio_integral(I_d1, f_d1, I_d2, f_d2):
	return (I_d2/I_d1)*(f_d1/f_d2)

def alkDenRatio_fitted(c_d1, g_d1, f_d1, c_d2, g_d2, f_d2):
	return (c_d2/c_d1)*(f_d1/f_d2)*(g_d1/g_d2)

'''----------------------------------------Main----------------------------------------
'''
if updates:
	print()
	print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
	print()

pathChecker(listOfPaths)

############ Gather the Data

data = np.loadtxt( Path(sys.argv[1]) )
file = str(sys.argv[1]).split("\\")[len(sys.argv[1].split("\\"))-1].split("/")[len(sys.argv[1].split("/")) - 1].split(".")[0]
cellName = file.split("_")[0]
runNumber = file.split("_")[1]
temp = re.sub('C', '', file.split("_")[2])
sweepNumber = file.split("_")[3]

fitSelect = int( sys.argv[2] )
fitUsed = ["Int", "Lor", "pV"][fitSelect]

if "kappa" in file.lower():
	in_FWHM = 16.0
	FWHM_LB = 0
	FWHM_HB = 30
	chi2_threshold_init = 20
	chi2_threshold = 20
else:
	in_FWHM = 119.0
	FWHM_LB = 50
	FWHM_HB = 225
	chi2_threshold_init = 50
	chi2_threshold = 50

### Split data into signal, wlm, errors, etc
ref = data[ :,1 ]
pri = data[ :,2 ]
wlm = data[ :,3 ]
ratio = np.log(pri/ref)

### Initialize things that actually need to be initialized
included = [False for _ in range(4)]
lines = ["RbD1", "RbD2", "KD1D2"]
wlmDict = {"RbD1":[], "RbD2":[], "KD1D2":[]}
ratioDict = {"RbD1":[], "RbD2":[], "KD1D2":[]}
exclDict = {"RbD1":0, "RbD2":0, "KD1D2":0}
center_expected = [377107.4, 384230.4281, 389286.1, 391016.2]

monitor([f"Analyzing {file} ({fitSelect}) {fitUsed}"])
### Sort by line, remove bad data points (large error)
error = np.abs(data[ :,5 ])
for idx, val in enumerate(wlm):
	if ratio[idx] != 0:
		if (val<377808):
			if (abs(error[idx]/ratio[idx]) < allowedError):
				if val>377147:
					included[0] = True
				wlmDict["RbD1"].append(val)
				ratioDict["RbD1"].append(ratio[idx])
			else:
				exclDict["RbD1"] += 1
		elif (val<385243):
			if (abs(error[idx]/ratio[idx]) < allowedError):
				if val>384270:
					included[1] = True
				wlmDict["RbD2"].append(val)
				ratioDict["RbD2"].append(ratio[idx])
			else:
				exclDict["RbD2"] += 1
		else:
			if (abs(error[idx]/ratio[idx]) < allowedError):
				if (val>389326):
					included[2] = True
				if (val>391056):
					included[3] = True
				wlmDict["KD1D2"].append(val)
				ratioDict["KD1D2"].append(ratio[idx])
			else:
				exclDict["RbD1"] += 1

### Begin plotting any diagnostic plots
if plot_diagnostic:
	plt.figure(file + "_Reference")
	plt.plot( wlm, ref, color='b', marker=',', ls='' )
	plt.figure(file + "_Primary")
	plt.plot( wlm, pri, color='b', marker=',', ls='' )

del ref
del pri
del wlm
del ratio

if updates:
	print("\n-----Number of Excluded Points-----\n")
	print("RbD1 Line:\t" + str(exclDict["RbD1"]) + " of 1256 or " + str( np.round( 100.0*exclDict["RbD1"]/1256, 3 )) + "%" )
	print("RbD2 Line:\t" + str(exclDict["RbD2"]) + " of 2024 or " + str( np.round( 100.0*exclDict["RbD2"]/2024, 3 )) + "%" )
	print("KD1D2 Line:\t" + str(exclDict["KD1D2"]) + " of 3702 or " + str( np.round( 100.0*exclDict["KD1D2"]/3702, 3 )) + "%\n")

# Start the main plots
plt.figure(file)
plt.suptitle( cellName + " at " + temp + t + "C")
plt.xlabel( "Frequency (GHz)" )
plt.ylabel( "LN(Primary:Reference)" )

dictOfResultsDicts = {}

included[1] = True

############ MEAT AND POTATOES ######################
for i, v in enumerate(lines):
	thisLine = v
	best_vals = []
	yFit = []
	x = np.asarray(wlmDict[thisLine])
	y = np.asarray(ratioDict[thisLine])
	
	chi2 = 9999
	step_c0 = in_step_c0
	step_c1 = in_step_c1
	chi2_min = chi2
	c0_minMarker = 9999
	in_c0 = init_c0
	in_c1 = init_c1
	chi2_threshold = chi2_threshold_init

	pVal = 0
	
	monitor([f"\tLine: {thisLine}"])
	flag = False
	if i==2:
		if included[i]:
			if fitSelect == 0:
				halfIdx = int(len(x) - 1)
				for idx, val in enumerate(x):
					if (val>390283):
						halfIdx = idx
						break
				# Gather data for integrated solution
				best_vals = calcIntegral(x[:halfIdx], y[:halfIdx]) ### -------------------------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
				d1 = {"FWHM":best_vals[3], "center":best_vals[2], "c0":0, "c1":0, "m":0, "b":0, "integral":best_vals[0], "chi2":0, "eta":0, "maxAtt":best_vals[1], "FWHM_std":0, "center_std":0}
				if included[3]:
					best_vals = calcIntegral(x[halfIdx:], y[halfIdx:]) ### ------------------------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
					d2 = {"FWHM":best_vals[3], "center":best_vals[2], "c0":0, "c1":0, "m":0, "b":0, "integral":best_vals[0], "chi2":0, "eta":0, "maxAtt":best_vals[1], "FWHM_std":0, "center_std":0}
			elif fitSelect == 1:

				while (abs(chi2) > chi2_goal):
					integralController = 0
					iterator = 1
					chi2_min_last = chi2_min
					chi2_min = 9999
					c0_minMarker_last = c0_minMarker
					while iterator < iterator_max:

						if flag:
							step = step_c1
							in_c1 += iterator*step
						else:
							step = step_c0
							in_c0 += iterator*step

						init_vals = [in_c0, in_c1, center_expected[2], in_FWHM, in_c0, in_c1, center_expected[3], in_FWHM, in_m, in_b]
						lb = [-np.inf for _ in range(10)]
						hb = [np.inf for _ in range(10)]

						lb[3], lb[7] = (FWHM_LB for _ in range(2))
						hb[3], hb[7] = (FWHM_HB for _ in range(2))
						hb[0], hb[4] = (0 for _ in range(2))

						lb[1], lb[5] = (0 for _ in range(2))
						hb[1], hb[5] = (1 for _ in range(2))
						try:
							best_vals, covar = curve_fit(K_lor, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### ----------------------------------> Returns [D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b]
							yFit = K_lor( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9])
							chi2, pVal = stats.chisquare(y, f_exp=yFit)
						except:
							pass

						iterator += 1
						if abs(chi2) < chi2_min:
							chi2_min = abs(chi2)
							c0_minMarker = (best_vals[0] + best_vals[4])/2

					if (abs(chi2) <= chi2_goal):
						break
					elif ((abs(chi2_min - chi2_min_last) < acceptableDiff) and (abs(chi2) < chi2_threshold)):
						break
					elif (abs(chi2_min) >= abs(chi2_min_last)):
						in_c0 = -5000*rnum.random()
						in_c1 = rnum.random()
						integralController = 0
						chi2_threshold += 1
					else:
						integralController += (chi2_min - chi2_min_last)
						step_c0/= gran_c0
						in_c0 = c0_minMarker - ((iterator_max/2.0)*step_c0)

					monitor([f"in_c0: {in_c0}", f"in_c1: {in_c1}"])
					monitor([f"\t\tLine: {thisLine} --> chi2: {np.round(chi2, 5)}\tchi2_thresh / chi2: {np.round(abs(chi2_threshold/chi2), 3)}"])

				maxAtt = K_lor( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9])
				d1 = {"FWHM":best_vals[3], "center":best_vals[2], "c0":best_vals[0], "c1":best_vals[1], "m":best_vals[8], "b":best_vals[9], "integral":0, "chi2":chi2, "eta":0, "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[3][3]), "center_std":np.sqrt(covar[2][2])}
				if included[3]:
					maxAtt = K_lor( best_vals[6], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9])
					d2 = {"FWHM":best_vals[7], "center":best_vals[6], "c0":best_vals[4], "c1":best_vals[5], "m":best_vals[8], "b":best_vals[9], "integral":0, "chi2":chi2, "eta":0, "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[7][7]), "center_std":np.sqrt(covar[6][6])}
			elif fitSelect == 2:
				
				while (abs(chi2) > chi2_goal):
					integralController = 0
					iterator = 1
					chi2_min_last = chi2_min
					chi2_min = 9999
					c0_minMarker_last = c0_minMarker
					while iterator < iterator_max:

						if flag:
							step = step_c1
							in_c1 += iterator*step
						else:
							step = step_c0
							in_c0 += iterator*step

						init_vals = [-1000, in_c1, center_expected[2], in_FWHM, in_eta, in_c0, in_c1, center_expected[3], in_FWHM, in_eta, in_m, in_b]
						lb = [-np.inf for _ in range(12)]
						hb = [np.inf for _ in range(12)]

						lb[4], lb[9] = (0 for _ in range(2))
						hb[4], hb[9] = (1 for _ in range(2))
						lb[3], lb[8] = (FWHM_LB for _ in range(2))
						hb[3], hb[8] = (FWHM_HB for _ in range(2))
						hb[0], hb[5] = (0 for _ in range(2))

						lb[1], lb[6] = (0 for _ in range(2))
						hb[1], hb[6] = (1 for _ in range(2))

						try:
							best_vals, covar = curve_fit(K_pV, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### ----------------------------------> Returns [D1c0, D1c1, D1c2, D1c3, D1eta, D2c0, D2c1, D2c2, D2c3, D2eta, m, b]
							yFit = K_pV( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11])
							chi2, pVal = stats.chisquare(y, f_exp=yFit)
						except:
							pass

						iterator += 1
						if abs(chi2) < chi2_min:
							chi2_min = abs(chi2)
							c0_minMarker = (best_vals[0] + best_vals[5])/2

					if (abs(chi2) <= chi2_goal):
						break
					elif ((abs(chi2_min - chi2_min_last) < acceptableDiff) and (abs(chi2) < chi2_threshold)):
						break
					elif (abs(chi2_min) >= abs(chi2_min_last)):
						in_c0  = -5000*rnum.random()
						in_c1 = rnum.random()
						integralController = 0
						chi2_threshold += 1
					else:
						integralController += (chi2_min - chi2_min_last)
						step_c0/= gran_c0
						in_c0 = c0_minMarker - ((iterator_max/2.0)*step_c0)

					monitor([f"in_c0: {in_c0}", f"in_c1: {in_c1}"])
					monitor([f"\t\tLine: {thisLine} --> chi2: {np.round(chi2, 5)}\tchi2_thresh / chi2: {np.round(abs(chi2_threshold/chi2), 3)}"])

				maxAtt = K_pV( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11])
				d1 = {"FWHM":best_vals[3], "center":best_vals[2], "c0":best_vals[0], "c1":best_vals[1], "m":best_vals[10], "b":best_vals[11], "integral":0, "chi2":chi2, "eta":best_vals[4], "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[3][3]), "center_std":np.sqrt(covar[2][2])}
				if included[3]:
					maxAtt = K_pV( best_vals[7], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11])
					d2 = {"FWHM":best_vals[8], "center":best_vals[7], "c0":best_vals[5], "c1":best_vals[6], "m":best_vals[10], "b":best_vals[11], "integral":0, "chi2":chi2, "eta":best_vals[4], "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[8][8]), "center_std":np.sqrt(covar[7][7])}
			else:
				print("NOT A VALID FIT")
				sys.exit()
			dictOfResultsDicts["KD1"] = copy.deepcopy(d1)
			if included[3]:
				dictOfResultsDicts["KD2"] = copy.deepcopy(d2)
	else:
		if included[i]:
			int_Cen = center_expected[i]
			if fitSelect == 0:
				best_vals = calcIntegral(x, y) ### ------------------------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
				d = {"FWHM":best_vals[3], "center":best_vals[2], "c0":0, "c1":0, "m":0, "b":0, "integral":best_vals[0], "chi2":0, "eta":0, "maxAtt":best_vals[1], "FWHM_std":0, "center_std":0}
			elif fitSelect == 1:
				while (abs(chi2) > chi2_goal):
					integralController = 0
					iterator = 1
					chi2_min_last = chi2_min
					chi2_min = 9999
					c0_minMarker_last = c0_minMarker
					while iterator < iterator_max:				

						if flag:
							step = step_c1
							in_c1 += iterator*step
						else:
							step = step_c0
							in_c0 += iterator*step

						init_vals = [in_c0, in_c1, int_Cen, in_FWHM, in_m, in_b]
						lb = [-np.inf for _ in range(6)]
						hb = [np.inf for _ in range(6)]
						lb[3] = FWHM_LB
						hb[3] = FWHM_HB
						hb[0] = 0

						lb[1] = 0
						hb[1] = 1

						try:
							best_vals, covar = curve_fit(Rb_lor, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### ------------> Returns [c0, c1, c2, c3, m, b]
							yFit = Rb_lor( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5])
							chi2, pVal = stats.chisquare(y, f_exp=yFit)
						except:
							pass

						iterator += 1
						if abs(chi2) < chi2_min:
							chi2_min = abs(chi2)
							c0_minMarker = best_vals[0]

					if (abs(chi2) <= chi2_goal):
						break
					elif ((abs(chi2_min - chi2_min_last) < acceptableDiff) and (abs(chi2) < chi2_threshold)):
						break
					elif (abs(chi2_min) >= abs(chi2_min_last)):
						in_c0 = -5000*rnum.random()
						in_c1 = rnum.random()
						integralController = 0
						chi2_threshold += 1
					else:
						integralController += (chi2_min - chi2_min_last)
						step_c0/= gran_c0
						in_c0 = c0_minMarker - ((iterator_max/2.0)*step_c0)

					monitor([f"in_c0: {in_c0}", f"in_c1: {in_c1}"])
					monitor([f"\t\tLine: {thisLine} --> chi2: {np.round(chi2, 5)}\tchi2_thresh / chi2: {np.round(abs(chi2_threshold/chi2), 3)}"])
						
				maxAtt = Rb_lor( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5])
				d = {"FWHM":best_vals[3], "center":best_vals[2], "c0":best_vals[0], "c1":best_vals[1], "m":best_vals[4], "b":best_vals[5], "integral":0, "chi2":chi2, "eta":0, "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[3][3]), "center_std":np.sqrt(covar[2][2])}
			elif fitSelect == 2:
				while (abs(chi2) > chi2_goal):
					integralController = 0
					iterator = 1
					chi2_min_last = chi2_min
					chi2_min = 9999
					c0_minMarker_last = c0_minMarker
					while iterator < iterator_max:

						if flag:
							step = step_c1
							in_c1 += iterator*step
						else:
							step = step_c0
							in_c0 += iterator*step

						init_vals = [in_c0, in_c1, int_Cen, in_FWHM, in_eta, in_m, in_b]
						lb = [-np.inf for _ in range(7)]
						hb = [np.inf for _ in range(7)]
						lb[4] = 0
						hb[4] = 1	
						lb[3] = FWHM_LB
						hb[3] = FWHM_HB
						hb[0] = 0

						lb[1] = 0
						hb[1] = 1

						try:
							best_vals, covar = curve_fit(Rb_pV, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### -------------> Returns [c0, c1, c2, c3, eta, m, b]
							yFit = Rb_pV( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6])
							chi2, pVal = stats.chisquare(y, f_exp=yFit)
						except:
							pass

						iterator += 1
						if abs(chi2) < chi2_min:
							chi2_min = abs(chi2)
							c0_minMarker = best_vals[0]

					if (abs(chi2) <= chi2_goal):
						break
					elif ((abs(chi2_min - chi2_min_last) < acceptableDiff) and (abs(chi2) < chi2_threshold)):
						break
					elif (abs(chi2_min) >= abs(chi2_min_last)):
						in_c0 = -5000*rnum.random()
						in_c1 = rnum.random()
						integralController = 0
						chi2_threshold += 1
					else:
						integralController += (chi2_min - chi2_min_last)
						step_c0/= gran_c0
						in_c0 = c0_minMarker - ((iterator_max/2.0)*step_c0)

					monitor([f"in_c0: {in_c0}", f"in_c1: {in_c1}"])
					monitor([f"\t\tLine: {thisLine} --> chi2: {np.round(chi2, 5)}\tchi2_thresh / chi2: {np.round(abs(chi2_threshold/chi2), 3)}"])
						
				maxAtt = Rb_pV( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6])
				d = {"FWHM":best_vals[3], "center":best_vals[2], "c0":best_vals[0], "c1":best_vals[1], "m":best_vals[5], "b":best_vals[6], "integral":0, "chi2":chi2, "eta":best_vals[4], "yFit":yFit, "x":x, "y":y, "maxAtt":maxAtt, "pVal":pVal, "FWHM_std":np.sqrt(covar[3][3]), "center_std":np.sqrt(covar[2][2])}
			else:
				print("NOT A VALID FIT")
				sys.exit()
			if i==0:
				dictOfResultsDicts["RbD1"] = copy.deepcopy(d)
			else:
				dictOfResultsDicts["RbD2"] = copy.deepcopy(d)
	if fitSelect != 0:
		plt.figure(file)
		plt.plot( x, y, color='r', marker='.', ms='.75', ls='', label=("Data") )
		try:
			plt.plot( x, yFit, color='k', marker=',', ls='-', lw='.5', label=("Fit") )
		except:
			pass


if (fitSelect != 0) and (plot_show or plot_save):
	exportPlots(plot_show, plot_save, plot_each) ### --> exportPlots(showPlots, savePlots, everyLine):

if fitSelect != 0:
	for k, d in dictOfResultsDicts.items():
		try:
			#del d["maxAtt"]
			#del d["c1"]
			#del d["m"]
			#del d["b"]
			del d["x"]
			del d["y"]
			del d["yFit"]
		except:
			pass
	
exportDict = {}

lines = ["RbD1", "RbD2", "KD1", "KD2"]

for idx, val in enumerate(included):
	if val:
		line = lines[idx]
		for k, v in dictOfResultsDicts[line].items():
			newKey = line + "_" + str(k)
			exportDict[newKey] = v

alkDenRatio = []
oscStrengths = [0.34231, 0.69577, 0.334, 0.672]
for K in range(2,4):
	for Rb in range(0,2):
		if included[Rb] and included[K]:
			if fitSelect != 0:
				rat = alkDenRatio_fitted(np.mean(dictOfResultsDicts[lines[Rb]]["c0"]), np.mean(dictOfResultsDicts[lines[Rb]]["FWHM"]), oscStrengths[Rb], np.mean(dictOfResultsDicts[lines[K]]["c0"]), np.mean(dictOfResultsDicts[lines[K]]["FWHM"]), oscStrengths[K])
				alkDenRatio.append(rat)
			else:
				rat = alkDenRatio_integral(np.mean(dictOfResultsDicts[lines[Rb]]["integral"]), oscStrengths[Rb], np.mean(dictOfResultsDicts[lines[K]]["integral"]), oscStrengths[K])
				alkDenRatio.append(rat)
if False: #alkDenRatio:
	entry = np.mean(alkDenRatio)
	exportDict["AlkDenRatio"] = entry

logger(exportDict, file, fitUsed)

s = f"{fitUsed}\t{cellName}\t{runNumber}\t{temp}\t{sweepNumber}"
for idx, val in enumerate(lines):
	if included[idx]:
		fwhm = dictOfResultsDicts[val]["FWHM"]
		fwhm_std = dictOfResultsDicts[val]["FWHM_std"]
		center = dictOfResultsDicts[val]["center"]
		center_std = dictOfResultsDicts[val]["center_std"]
		maxAtt = dictOfResultsDicts[val]["maxAtt"]
		eta = dictOfResultsDicts[val]["eta"]
		s += f"\n{val}\t{fwhm}\t{center}\t{maxAtt}\t{eta}"
if False: # alkDenRatio:
	entry = np.mean(alkDenRatio)
	s += f"\nAlk.Den.Ratio\t{entry}"
s += "\n\n"

f = "__pasteToExcel.txt"
os.chdir(analysisPath)
writer = open(f, "a")
writer.write(s)
writer.close()
os.chdir(homePath)

os.chdir(picklePath)
export_pickle(exportDict, "exportDict.pickle")
os.chdir(homePath)

print("Here's what you're exporting...")
for k, v in exportDict.items():
	print(f"\t{k}")

print("Here's what's in each dictionary...")

for idx, val in enumerate(included):
	if val:
		line = lines[idx]
		print(f"\t{line}")
		for k, v in dictOfResultsDicts[line].items():
			print(f"\t\t{k}")
		print()

if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()