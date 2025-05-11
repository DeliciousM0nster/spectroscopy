'''----------------------------------------Imports----------------------------------------
'''
import sys, os, re, pickle, copy
import numpy as np
from pathlib import Path
import scipy.stats as stats
import datetime as ddtt
import pandas as pd

t = u"\u00b0"

'''----------------------------------------Flags and Options----------------------------------------
'''
updates = True

'''----------------------------------------Functions----------------------------------------
'''
### Notes "Pull it out the brine!"
	# Purpose: Import a serialized file
	# Takes: string, name of file
	# Returns: array
def import_unpickle(fileIn):
	with open(fileIn, 'rb') as x:
		temp = pickle.load(x)
	return temp

### Notes "Dunk it in the brine!"
	# Purpose: Export a serialized file (read by "run_multiSpec.py")
	# Takes: n (thing being pickled), line (RbD1, RbD2, KD1D2)
	# Exports: Pickled file (usually of residuals) labelled with the line it's for
def export_pickle(n, file):
	with open(file, 'wb') as e:
		pickle.dump(n, e)

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

def alkDenRatio_integral(I_d1, f_d1, I_d2, f_d2):
	return (I_d2/I_d1)*(f_d1/f_d2)

def alkDenRatio_fitted(c_d1, g_d1, f_d1, c_d2, g_d2, f_d2):
	return (c_d2/c_d1)*(f_d1/f_d2)*(g_d1/g_d2)

def D(T):
	T += 273.15
	return 10**((4.402-4.312) - (4453-4040)/float(T))



'''----------------------------------------Main----------------------------------------
'''

if updates:
	print()
	print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
	print()

lineNames = ["RbD1", "RbD2", "KD1", "KD2"]


textFile = open(Path(Path.cwd(), "__AllData", "tableData.txt"),'w')

bigOldDict = import_unpickle(Path(Path.cwd(), "__AllData", "results_all_bestValsANDerrors.pickle"))

for cellName, run_LIST in bigOldDict.items():
	textFile.write(f"\n\n\n\n------{cellName}------\n\n")

	
	for run_IDX, temp_DICT in enumerate(run_LIST):
		if temp_DICT:
			firstFlag_temp = True
			for temp, line_LIST in temp_DICT.items():
				thisOutputLine = ""
				if firstFlag_temp:
					thisOutputLine += f"Run {run_IDX}"
					firstFlag_temp = False
				thisOutputLine += " & " + "\\ang{" + str(temp) + "} & "
				for line_IDX, fit_LIST in enumerate(line_LIST):
					if line_IDX > 0:
						thisOutputLine = " &  & "
					thisOutputLine += lineNames[line_IDX]
					text_fitted, text_integral = ("" for _ in range(2))
					for fit_IDX, data_DICT in enumerate(fit_LIST):
						if fit_IDX != 2:
							avg_FWHM, avg_center, avg_att, avg_eta = ([] for _ in range(4))
							err_FWHM, err_center, err_att, err_eta = (0 for _ in range(4))
						for dataSet_IDX in range(len(data_DICT["best_vals"])):
							print(cellName)
							print(thisOutputLine)
							#print(f"{cellName} - Run {run_IDX} - {temp}{t}C - fit:{fit_IDX} - {lineNames[line_IDX]}")
							avg_FWHM.append(data_DICT["best_vals"][dataSet_IDX][3])
							avg_center.append(data_DICT["best_vals"][dataSet_IDX][2])
							try:
								err_FWHM += data_DICT["covar"][dataSet_IDX][3]**2
								err_center += data_DICT["covar"][dataSet_IDX][2]**2
							except:
								pass
							if fit_IDX == 0:
								avg_att.append(data_DICT["best_vals"][dataSet_IDX][1])
							elif line_IDX > 1:

								kVals = []
								for i in range(3 + fit_IDX):
									kVals.append(line_LIST[2][fit_IDX]["best_vals"][dataSet_IDX][i])
								for i in range(len(line_LIST)):
									print(line_LIST[i][fit_IDX]["best_vals"][dataSet_IDX])
								kVals = kVals + line_LIST[3][fit_IDX]["best_vals"][dataSet_IDX]



								x = data_DICT["best_vals"][dataSet_IDX][2]
								if fit_IDX == 1:
									avg_att.append(K_lor(x, kVals[0], kVals[1], kVals[2], kVals[3], kVals[4], kVals[5], kVals[6], kVals[7], kVals[8], kVals[9]) - kVals[8]*x - kVals[9])
								else:
									avg_att.append(K_pV(x, kVals[0], kVals[1], kVals[2], kVals[3], kVals[4], kVals[5], kVals[6], kVals[7], kVals[8], kVals[9], kVals[10], kVals[11]) - kVals[10]*x - kVals[11])
									avg_eta.append(data_DICT["best_vals"][dataSet_IDX][4])
									err_eta += data_DICT["covar"][dataSet_IDX][4]**2



							elif fit_IDX == 1:
								RbVals = data_DICT["best_vals"][dataSet_IDX]
								avg_att.append(Rb_lor(RbVals[2], RbVals[0], RbVals[1], RbVals[2], RbVals[3], RbVals[4], RbVals[5]) - RbVals[2]*RbVals[4] - RbVals[5])
							else:
								RbVals = data_DICT["best_vals"][dataSet_IDX]
								avg_att.append(Rb_pV(RbVals[2], RbVals[0], RbVals[1], RbVals[2], RbVals[3], RbVals[4], RbVals[5], RbVals[6]) - RbVals[2]*RbVals[5] - RbVals[6])
								avg_eta.append(data_DICT["best_vals"][dataSet_IDX][4])
								err_eta += data_DICT["covar"][dataSet_IDX][4]**2
						if fit_IDX !=1:
							if len(avg_FWHM) > 0:
								out_FWHM = np.round(np.mean(avg_FWHM), 3)
								out_center = np.round(np.mean(avg_center), 2)
								out_att = np.round(np.mean(avg_att), 3)
								out_eta = np.round(np.mean(avg_eta), 3)
								out_err_FWHM = np.round(np.sqrt(err_FWHM), 3)
								out_err_center = np.round(np.sqrt(err_center), 3)
								out_err_eta = np.round(np.sqrt(err_eta), 3)
							else:
								out_FWHM, out_center, out_att = ("" for _ in range(3))
							if fit_IDX == 0:
								text_integral = f"{out_FWHM} & {out_center} & {out_att}"
							else:
								text_fitted = f"{out_FWHM}({out_err_FWHM}) & {out_center}({out_err_center}) & {out_att} & {out_eta}({out_err_eta})"
					thisOutputLine += " & " + text_fitted + " & " + text_integral + " \\\\ \\hline\n"
					textFile.write(thisOutputLine)
				## Alkali ratio stuff goes here
		textFile.write("\n\n")

							




							









'''
f = [0.34231, 0.69577, 0.334, 0.672] # [RbD1, RbD2, KD1, KD2]

temps = [130, 135, 140, 145]

data_130 = [ [-234.4716878, 125.1398186], [-516.9928371, 126.9210486], [-381.4129814, 86.93712276], [-1231.22815, 110.0036763] ]

data_135 = [ [-299.4250507, 118.8231534], [-726.6722723, 122.3243054], [-538.1324686, 82.56074289], [-1747.49290, 106.2993903] ]

data_140 = [ [-406.3908634, 111.2548422], [-1040.750037, 111.3562481], [-847.0664826, 78.16361868], [-2775.69042, 98.74222822]]

data_145 = [ [-502.3394667, 105.9440976], [-1334.373059, 104.3493128], [-1090.8113, 73.39801656], [-3965.558393, 97.20556681] ]

allData = [data_130, data_135, data_140, data_145]

finalNumbers = []

for IDX, DATA in enumerate(allData):
	allTheseRatios = []
	for D1 in range(0, 4, 2):
		for D2 in range(1, 5, 2):
			thisRatio = alkDenRatio_fitted(DATA[D1][0], DATA[D1][1], f[D1], DATA[D2][0], DATA[D2][1], f[D2])
			print(f"Ratio {D2}:{D1} at T = {130 + (IDX*5)}: {np.round(thisRatio, 3)}")
			allTheseRatios.append(thisRatio)
	print(f"\tRatio for T = {130 + (IDX*5)}: {np.round(np.mean(allTheseRatios), 3)} +/- {np.round(np.std(allTheseRatios), 3)}")
	finalNumbers.append(np.mean(allTheseRatios))

print()

lastOne = []

for IDX in range(4):
	print(f"D(235)/D({130 + (IDX*5)}) times data at T = {130 + (IDX*5)}: {finalNumbers[IDX]*D(235)/D(130 + (IDX*5))}")
	lastOne.append(finalNumbers[IDX]*D(235)/D(130 + (IDX*5)))

print(f"\nFinal Answer for T=235C: {np.round(np.mean(lastOne), 3)} +/- {np.round(np.std(lastOne), 3)}")


'''



if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()
