'''----------------------------------------Imports----------------------------------------
'''

import sys, re, os, gc
import numpy as np
import pickle
import scipy.stats as stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
import copy
import random as rnum
import datetime as dt
t = u"\u00b0"
pm = " " + u"\u00B1" + " "

from sympy.physics.wigner import wigner_6j
from sympy.physics.wigner import wigner_3j

'''----------------------------------------Flags and Options----------------------------------------
'''
tryCount = 0
allowedError = 1
updates = False
attemptsToFit = 5000000
plot_show = False
plot_save = True
plot_each = False
plot_diagnostic = False

homePath = Path.cwd()
picklePath = Path(homePath, "__AllData")
plotPath = Path(picklePath, "specAnal_Plots")
logPath = Path(picklePath, "specAnal_Logs")
listOfPaths = [plotPath, homePath, picklePath, logPath]

startTime = dt.datetime.now()
elapsed = startTime - dt.datetime.now()
breakGlassAt = 300

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
	if True:
		for i, v in enumerate(t):
			if i==0:
				print(f"\t--> {v}")
			else:
				print(f"\t    {v}")

def pathChecker(pList):
	for i, p in enumerate(pList):
		try:
			os.makedirs(p)
		except:
			pass

def passVibeCheck(best, covar, c, c_goal, c_min, c_min_last, c_passable, var, line):

	defThresh = c_passable/var/2.0
	tf_1 = (abs(c) <= c_goal)
	tf_2 = abs(c) < defThresh
	tf_3 = np.sqrt(covar[3][3]) < 10
	if line == "KD1D2":
		tf_4 = (np.sqrt(covar[3][3]) < 2)
	else:
		tf_4 = True
	return (tf_1 or tf_2) and tf_3 and tf_4


#####################

def Rb_lor( x, c0, c1, c2, c3, m, b):
	del1 = (x - c2 + 1.264887)
	del2 = (x - c2 - 1.770884)
	del3 = (x - c2 + 2.563005)
	del4 = (x - c2 - 4.271676)
	part1 = ((1 + (0.664*2*np.pi*c1*del1)) / (del1**2 + (c3/2)**2))
	part2 = ((1 + (0.664*2*np.pi*c1*del2)) / (del2**2 + (c3/2)**2))
	part3 = ((1 + (0.664*2*np.pi*c1*del3)) / (del3**2 + (c3/2)**2))
	part4 = ((1 + (0.664*2*np.pi*c1*del4)) / (del4**2 + (c3/2)**2))
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

def inVals_and_bounds(line, c0, c1, c3, lb_c3, hb_c3, fitType):
	c2 = [377107.4, 384230.4281, 389286.1, 391016.2]
	init_vals = [c0, c1, c2[line], c3]
	lb = [-np.inf, 0, -np.inf, lb_c3]
	hb = [0, 0.1, np.inf, hb_c3]
	if fitType == 2:
		init_vals.append(0.61)
		lb.append(0)
		hb.append(1)
	if line == 2:
		init_vals.extend([c0, c1, c2[3], c3])
		lb.extend([-np.inf, 0, -np.inf, lb_c3])
		hb.extend([0, 1, np.inf, hb_c3])
		if fitType == 2:
			init_vals.append(0.61)
			lb.append(0)
			hb.append(1)
	init_vals.extend([0.000000133, -0.67969])
	lb.extend([-np.inf, -np.inf])
	hb.extend([np.inf, np.inf])
	return (init_vals, lb, hb)

#####################

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
	global fitSelect
	plt.figure(file)
	if everyLine:
		plt.close()
		for idx in range(3):
			if included[idx]:
				if idx == 2:
					s_param = "D1 FWHM: " + str(np.round(dictOfResultsDicts["KD1"]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["err_FWHM"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts["KD1"]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["err_cen"], 3)) + " GHz"
					if included[3]:
						s_param += "\nD2 FWHM: " + str(np.round(dictOfResultsDicts["KD2"]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["err_FWHM"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts["KD2"]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["err_cen"], 3)) + " GHz"
					s_stats = "chi2: " + str(np.round(dictOfResultsDicts["KD1"]["chi2"], 3))
					s = s_param + "\n\n" + s_stats
					generatePlot("KD1D2", dictOfResultsDicts["KD1"]["x"], dictOfResultsDicts["KD1"]["y"], dictOfResultsDicts["KD1"]["yFit"], s)
				else:
					s_param = "FWHM: " + str(np.round(dictOfResultsDicts[lines[idx]]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts[lines[idx]]["err_FWHM"], 3)) + " GHz\nCenter: " + str(np.round(dictOfResultsDicts[lines[idx]]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts[lines[idx]]["err_cen"], 3)) + " GHz"
					s_stats = "chi2: " + str(np.round(dictOfResultsDicts[lines[idx]]["chi2"], 3))
					s = s_param + "\n\n" + s_stats
					generatePlot(lines[idx], dictOfResultsDicts[lines[idx]]["x"], dictOfResultsDicts[lines[idx]]["y"], dictOfResultsDicts[lines[idx]]["yFit"], s)
	else:
		s_c3 = "FWHM (GHz):"
		s_c2 = "Center:"
		s_stats = "Stats:"
		for idx in range(3):
			if included[idx]:
				if idx == 2:
					s_c3 += f"\n   KD1: " + str(np.round(dictOfResultsDicts["KD1"]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["err_FWHM"], 3))
					s_c2 += f"\n   KD1: " + str(np.round(dictOfResultsDicts["KD1"]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts["KD1"]["err_cen"], 3))
					if included[3]:
						s_c3 += f"\n   KD2: " + str(np.round(dictOfResultsDicts["KD2"]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["err_FWHM"], 3))
						s_c2 += f"\n   KD2: " + str(np.round(dictOfResultsDicts["KD2"]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts["KD2"]["err_cen"], 3))
					s_stats += f"\n   K-Lines: [chi2: " + str(np.round(abs(dictOfResultsDicts["KD1"]["chi2"]), 4)) + "]"
				else:
					s_c3 += f"\n   {lines[idx]}: " + str(np.round(dictOfResultsDicts[lines[idx]]["val_FWHM"], 3)) + pm + str(np.round(dictOfResultsDicts[lines[idx]]["err_FWHM"], 3))
					s_c2 += f"\n   {lines[idx]}: " + str(np.round(dictOfResultsDicts[lines[idx]]["val_cen"], 3)) + pm + str(np.round(dictOfResultsDicts[lines[idx]]["err_cen"], 3))
					s_stats += f"\n   {lines[idx]}: [chi2: " + str(np.round(abs(dictOfResultsDicts[lines[idx]]["chi2"]), 4)) + "]"

		xpos = plt.xlim()[0] + 0.1*(plt.xlim()[1] - plt.xlim()[0])
		ypos = plt.ylim()[0] + 0.1*(plt.ylim()[1] - plt.ylim()[0])
		s = s_c3 + "\n\n" + s_c2 + "\n\n" + s_stats
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
	for k, v in d.items():
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

def errorLogger(file, fit, fitName, isErr, message):
	entry = f"\n{file} {fit} {fitName} ---> "
	if isErr:
		entry += "emergency exited"
	else:
		entry += f"{message}"
	entry += "\n"
	errorFile = Path(logPath, "emergencyExit.txt")
	errorLog = open(errorFile, "a")
	errorLog.write(entry)
	errorLog.close()

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

############ Gather the Data #############

data = np.loadtxt( Path(sys.argv[1]) )
file = str(sys.argv[1]).split("\\")[len(sys.argv[1].split("\\"))-1].split("/")[len(sys.argv[1].split("/")) - 1].split(".")[0]
cellName = file.split("_")[0]
runNumber = file.split("_")[1]
temp = re.sub('C', '', file.split("_")[2])
sweepNumber = file.split("_")[3]

fitSelect = int( sys.argv[2] )
fitUsed = ["Int", "Lor", "pV"][fitSelect]

### Split data into signal, wlm, errors, etc
ref = data[ :,1 ]
pri = data[ :,2 ]
wlm = data[ :,3 ]
ratio = np.log(pri/ref)

### Initialize things that actually need to be initialized
included = [False for _ in range(4)]
lines = ["RbD1", "RbD2", "KD1", "KD2"]
wlmDict = {"RbD1":[], "RbD2":[], "KD1D2":[]}
ratioDict = {"RbD1":[], "RbD2":[], "KD1D2":[]}
exclDict = {"RbD1":0, "RbD2":0, "KD1D2":0}

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
				exclDict["KD1D2"] += 1

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


optimal_chi2, optimal_c0, optimal_c1 = ([] for _ in range(3))


############ MEAT AND POTATOES ######################
for i in range(3):
	if i==2:
		thisLine = "KD1D2"
	else:
		thisLine = lines[i]
	best_vals = []
	yFit = []
	x = np.asarray(wlmDict[thisLine])
	y = np.asarray(ratioDict[thisLine])
	
	if "kappa" in file.lower(): cellSpecificVariables = [20, 12, 0, 30]
	else: cellSpecificVariables = [50, 119, 50, 225]

	pVal = 0
	chi2 = 999
	chi2_min = 9999
	chi2_min_last = 99999
	chi2_passable = cellSpecificVariables[0]
	chi2_goal = 1

	this_c0 = -4000

	c0_minMarker_forChi2 = 9999
	step_c0 = -2*cellSpecificVariables[0]
	in_c0 = this_c0 + (20*cellSpecificVariables[0]*(rnum.random() - 0.5))
	in_c1 = 0.001*rnum.random()

	in_c3 = cellSpecificVariables[1]
	lb_c3 = cellSpecificVariables[2]
	hb_c3 = cellSpecificVariables[3]

	acceptableDiff = chi2_goal

	monitor([f"\tLine: {thisLine}"])

	best_vals, covar = ([] for _ in range(2))

	if included[i]:
		if fitSelect == 0:
			if i == 2:
				halfIdx = int(len(x) - 1)
				for idx, val in enumerate(x):
					if (val>390283):
						halfIdx = idx
						break
				# Gather data for integrated solution
				best_vals = calcIntegral(x[:halfIdx], y[:halfIdx]) ### -------------------------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
				dictOfResultsDicts["KD1"] = {"val_FWHM":best_vals[3], "val_cen":best_vals[2], "val_maxAtt":best_vals[1], "val_c0":best_vals[0], "chi2":0, "eta":0}
				if included[3]:
					best_vals = calcIntegral(x[halfIdx:], y[halfIdx:]) ### ------------------------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
					dictOfResultsDicts["KD2"] = {"val_FWHM":best_vals[3], "val_cen":best_vals[2], "val_maxAtt":best_vals[1], "val_c0":best_vals[0], "chi2":0, "eta":0}
			else:
				best_vals = calcIntegral(x, y) ### ---------------------------------------------> Returns [integral, peak_y, peak_x, FWHM]
				dictOfResultsDicts[lines[i]] = {"val_FWHM":best_vals[3], "val_cen":best_vals[2], "val_maxAtt":best_vals[1], "val_c0":best_vals[0], "chi2":0, "eta":0}
		else:
			while (abs(chi2) > chi2_goal):
				if elapsed > dt.timedelta(seconds=breakGlassAt):
					print(f"---> Emergency Break on File: {file}")
					errorLogger(file, fitSelect, fitUsed, True, "")
					errorLogger(file, fitSelect, fitUsed, False, f"Optimal Settings for chi2 {np.round(np.mean( optimal_chi2 ), 3)} ({np.round(np.median( optimal_chi2 ), 3)}) +/- {np.round(np.std( optimal_chi2 ), 3)} appear to be:\n\tc0 = {np.mean( optimal_c0 )} ({np.median( optimal_c0 )}) +/- {np.std( optimal_c0 )}\n\tc1 = {np.mean( optimal_c1 )} ({np.median( optimal_c1 )}) +/- {np.mean( optimal_c1 )}")
					sys.exit()
				else:
					elapsed = dt.datetime.now() - startTime
				
				intCont = 0
				chi2_min_last = 0

				for iterator in range(500):				

					init_vals, lb, hb = inVals_and_bounds(i, in_c0, in_c1, in_c3, lb_c3, hb_c3, fitSelect)
					try:
						if i == 2:
							if fitSelect == 1:
								best_vals, covar = curve_fit(K_lor, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### ------------> Returns [D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b]
								yFit = K_lor( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9])
								maxAtt_D1 = K_lor(best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9]) - (best_vals[8]*best_vals[2] + best_vals[9])
								if included[3]:
									maxAtt_D2 = K_lor(best_vals[6], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9]) - (best_vals[8]*best_vals[6] + best_vals[9])
							elif fitSelect == 2:	
								best_vals, covar = curve_fit(K_pV, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### -------------> Returns [D1c0, D1c1, D1c2, D1c3, D1eta, D2c0, D2c1, D2c2, D2c3, D2eta, m, b]
								yFit = K_pV( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11])
								maxAtt_D1 = K_pV(best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11]) - (best_vals[10]*best_vals[2] + best_vals[11])
								if included[3]:
									maxAtt_D2 = K_pV(best_vals[7], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6], best_vals[7], best_vals[8], best_vals[9], best_vals[10], best_vals[11]) - (best_vals[10]*best_vals[7] + best_vals[11])
						else:
							if fitSelect == 1:
								best_vals, covar = curve_fit(Rb_lor, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### ------------> Returns [c0, c1, c2, c3, m, b]
								yFit = Rb_lor( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5])
								maxAtt = Rb_lor( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5]) - (best_vals[4]*best_vals[2] + best_vals[5])
							elif fitSelect == 2:	
								best_vals, covar = curve_fit(Rb_pV, x, y, p0=init_vals, bounds=(lb, hb), maxfev=attemptsToFit) ### -------------> Returns [c0, c1, c2, c3, eta, m, b]
								yFit = Rb_pV( x, best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6])
								maxAtt = Rb_pV( best_vals[2], best_vals[0], best_vals[1], best_vals[2], best_vals[3], best_vals[4], best_vals[5], best_vals[6]) - (best_vals[5]*best_vals[2] + best_vals[6])
						chi2, pVal = stats.chisquare(y, f_exp=yFit)
						errorLogger(file, fitSelect, fitUsed, False, f"Fit success\n\tchi2 = {chi2}\n\tc_0 = {best_vals[0]}\n\tc_1 = {best_vals[1]}")
						if chi2 < 1:
							optimal_chi2.append(chi2)
							optimal_c0.append(best_vals[0])
							optimal_c1.append(best_vals[1])
					except:
						pass

					### Vibe Check ###
					if passVibeCheck(best_vals, covar, chi2, chi2_goal, chi2_min, chi2_min_last, chi2_passable, cellSpecificVariables[0], thisLine):
						break

					if abs(chi2) < chi2_min:
						chi2_min_last = chi2_min
						chi2_min = abs(chi2)
						c0_minMarker_forChi2 = best_vals[0]
					in_c0 -= step_c0*iterator
					if (in_c0 < -10000):
						in_c0 = this_c0 + (20*cellSpecificVariables[0]*(rnum.random() - 0.5))
						in_c1 = 0.001*rnum.random()

				### Vibe Check ###
				if passVibeCheck(best_vals, covar, chi2, chi2_goal, chi2_min, chi2_min_last, chi2_passable, cellSpecificVariables[0], thisLine):
					break

				if abs(chi2) < chi2_min:
					chi2_min_last = chi2_min
					chi2_min = abs(chi2)
					c0_minMarker_forChi2 = best_vals[0]
					intCont += (chi2_min - chi2_min_last)
					step_c0/= 5.0
					in_c0 = c0_minMarker_forChi2 - ((10)*step_c0)
				else:
					chi2_passable += 5*cellSpecificVariables[0]/10.0
					intCont = 0
					in_c0 = this_c0 + (20*cellSpecificVariables[0]*(rnum.random() - 0.5))
					in_c1 = 0.001*rnum.random()

				if (abs(chi2) > chi2_goal):
					monitor([f"{file} ({fitUsed}): {thisLine}", f"\tin_c0: {in_c0}", f"\tin_c1: {in_c1}", f"\tchi2:           {np.round(chi2, 5)}", f"\tchi2_threshold: {chi2_passable/cellSpecificVariables[0]/2}"])
				



			if i == 2:	
				eta_D1 = 0
				eta_D1_std = 0
				eta_D2 = 0
				eta_D2_std = 0
				FWHM_D1 = best_vals[3]
				FWHM_D1_std = np.sqrt(covar[3][3])
				FWHM_D2 = best_vals[7]
				FWHM_D2_std = np.sqrt(covar[7][7])
				c0_D1 = best_vals[0]
				c0_D1_std = np.sqrt(covar[0][0])
				c0_D2 = best_vals[4]
				c0_D2_std = np.sqrt(covar[4][4])
				c2_D1 = best_vals[2]
				c2_D1_std = np.sqrt(covar[2][2])
				c2_D2 = best_vals[6]
				c2_D2_std = np.sqrt(covar[6][6])
				if fitSelect == 2:
					eta_D1 = best_vals[4]
					eta_D1_std = np.sqrt(covar[4][4])
					eta_D2 = best_vals[9]
					eta_D2_std = np.sqrt(covar[9][9])
					FWHM_D2 = best_vals[8]
					FWHM_D2_std = np.sqrt(covar[8][8])
					c0_D2 = best_vals[5]
					c0_D2_std = np.sqrt(covar[5][5])
					c2_D2 = best_vals[7]
					c2_D2_std = np.sqrt(covar[7][7])

				d1 = {"val_c0":c0_D1, "err_c0":c0_D1_std, "val_FWHM":FWHM_D1, "err_FWHM":FWHM_D1_std, "val_cen":c2_D1, "err_cen":c2_D1_std, "val_maxAtt":maxAtt_D1, "val_eta":eta_D1, "err_eta":eta_D1_std, "x":x, "y":y, "yFit":yFit, "chi2":chi2}
				dictOfResultsDicts["KD1"] = copy.deepcopy(d1)
				if included[3]:
					d2 = {"val_c0":c0_D2, "err_c0":c0_D2_std, "val_FWHM":FWHM_D2, "err_FWHM":FWHM_D2_std, "val_cen":c2_D2, "err_cen":c2_D2_std, "val_maxAtt":maxAtt_D2, "val_eta":eta_D2, "err_eta":eta_D2_std, "x":x, "y":y, "yFit":yFit, "chi2":chi2}
					dictOfResultsDicts["KD2"] = copy.deepcopy(d2)

			else:
				eta = 0
				eta_std = 0
				FWHM = best_vals[3]
				FWHM_std = np.sqrt(covar[3][3])
				c0 = best_vals[0]
				c0_std = np.sqrt(covar[0][0])
				c2 = best_vals[2]
				c2_std = np.sqrt(covar[2][2])

				if fitSelect == 2:
					eta = best_vals[4]
					eta_std = np.sqrt(covar[4][4])
				d = {"val_c0":c0, "err_c0":c0_std, "val_FWHM":FWHM, "err_FWHM":FWHM_std, "val_cen":c2, "err_cen":c2_std, "val_maxAtt":maxAtt, "val_eta":eta, "err_eta":eta_std, "x":x, "y":y, "yFit":yFit, "chi2":chi2}
				dictOfResultsDicts[lines[i]] = copy.deepcopy(d)

			plt.figure(file)
			plt.plot( x, y, color='r', marker='.', ms='.75', ls='', label=("Data") )
			try:
				plt.plot( x, yFit, color='k', marker=',', ls='-', lw='.5', label=("Fit") )
			except:
				pass

if (fitSelect != 0) and (plot_show or plot_save):
	exportPlots(plot_show, plot_save, plot_each) ### --> exportPlots(showPlots, savePlots, everyLine):

for k, d in dictOfResultsDicts.items():
	try:
		del d["x"]
		del d["y"]
		del d["yFit"]
	except:
		pass

#K_lor( x, D1c0, D1c1, D1c2, D1c3, D2c0, D2c1, D2c2, D2c3, m, b):
#K_pV( x, D1c0, D1c1, D1c2, D1c3, D1eta, D2c0, D2c1, D2c2, D2c3, D2eta, m, b):

logger(dictOfResultsDicts, file, fitUsed)

os.chdir(picklePath)
export_pickle(dictOfResultsDicts, "exportedDict_specAnalysis.pickle")
os.chdir(homePath)

#print(dictOfResultsDicts)


if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()