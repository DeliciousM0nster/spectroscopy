'''----------------------------------------Imports----------------------------------------
'''

import sys
import os
import glob
import pickle
import numpy as np
import math as m
import re
import pandas as pd

from datetime import datetime
from datetime import timedelta
from pathlib import Path
import copy


'''----------------------------------------Flags and Options----------------------------------------
'''
timeElapsed = []
homePath = Path.cwd()
dataPath = Path(homePath, "__AllData", "__FullMonty")
picklePath = Path(homePath, "__AllData")
errorPath = Path(picklePath, "runGather_Logs")
listOfPaths = [homePath, dataPath, picklePath, errorPath]

'''----------------------------------------Functions----------------------------------------
'''
### Notes
	# Purpose: Import a serialized file
	# Takes: string, name of file
	# Returns: array
def importAndUnpickle(fileIn):
	temp = []
	with open(fileIn, 'rb') as x:
		temp = pickle.load(x)
	return temp

### Notes "Dunk it in the brine!"
	# Purpose: Export a serialized file (read by "run_multiSpec.py")
	# Takes: n (thing being pickled), line (RbD1, RbD2, KD1D2)
	# Exports: Pickled file (usually of residuals) labelled with the line it's for
def export_toPickle(pPath, n, message):
	cucumber = "exportedDict_gatherTargetInfo" + message + ".pickle"
	jar = Path(pPath, cucumber)
	with open(jar, 'wb') as e:
		pickle.dump(n, e)
	return str(jar)

def stamp():
	x = datetime.now()
	startDate = re.sub('-', '', str(x.date()))
	startTime = str(x.time()).split(".")[0]
	startTime = startTime.split(":")[0] + "h" + startTime.split(":")[1] + "m" + startTime.split(":")[2] + "s"
	timeElapsed.append(x)
	return f"D{startDate}_T{startTime}"

def errorLogger(f):
	if f == "tar":
		entry = "\n\n\n\n"
	elif f == "run":
		entry = "\n\n"
	elif f == "tmp":
		entry = "---\n"
	else:
		entry = f"{f} ---> exited early\n"
	errorFile = Path(errorPath, "multiErrorLog.txt")
	errorLog = open(errorFile, "a")
	errorLog.write(entry)
	errorLog.close()

def pathChecker(pList):
	for i, p in enumerate(pList):
		try:
			os.makedirs(p)
		except:
			pass

def monitor(t):
	for i, v in enumerate(t):
		print(f"\t--> {v}")

def alkDenRatio_integral(I_d1, f_d1, I_d2, f_d2):
	return (I_d2/I_d1)*(f_d1/f_d2)

def alkDenRatio_fitted(c_d1, g_d1, f_d1, c_d2, g_d2, f_d2):
	return (c_d2/c_d1)*(f_d1/f_d2)*(g_d1/g_d2)

def D(T):
	T += 273.15
	return 10**((4.402-4.312) - (4453-4040)/float(T))

def slapDash(low, val_FWHM, val_cen, val_att, val_eta, err_FWHM, err_cen, err_eta):
	dictOfLines = {}
	for line_idx, line_NAME in enumerate(line_nameType):
		if val_FWHM[line_idx]:
			fwhm = str(np.round(np.mean(val_FWHM[line_idx]), 3))
			cen = str(np.round(np.mean(val_cen[line_idx]), 2))
			att = str(np.round(np.mean(val_att[line_idx]), 3))
			if low > 0:
				fwhm += "(" + str(np.round(np.sqrt(err_FWHM[line_idx]), 3)) + ")"
				cen += "(" + str(np.round(np.sqrt(err_cen[line_idx]), 2)) + ")"
				eta = str(np.round(np.mean(val_eta[line_idx]), 3))
				eta += "(" + str(np.round(np.sqrt(err_eta[line_idx]), 3)) + ")"
			dictOfLines[line_NAME] = f"{line_NAME} & {fwhm} & {cen} & {att}"
			if low > 0:
				dictOfLines[line_NAME] += f" & {eta}"
	return dictOfLines

def cut_print(results, low):
	fitSelect = ["(Integrated)", "(Fitted)"]
	highSets = {"Brianna":[1], "Florence":[1, 2], "Fulla":[1, 2, 3, 4, 5], "Kappa1":[2], "Kappa2":[1], "Kappa3":[1], "Kappa4":[1, 2], "Noah":[1], "SandyII":[1, 2], "Savior":[1], "Wayne":[1]}
	for target, dictOfRuns in results.items():
		for runNum, dictOfTemps in dictOfRuns.items():
			out_table = f"\n\n\n\n------{target} {fitSelect[low]}------\n\nRun {runNum} & "
			out_excel = f"\n\n\n\n------{target} {fitSelect[low]}------\n\nRun {runNum} & "
			allRatios = []
			allRatios_err = 0
			for temp, listOfResults in dictOfTemps.items():
				firstLine = True
				firstOfRun = True
				for line, string in listOfResults[0][0].items():
					if firstOfRun:
						out_table += " \\ang{" + str(temp) + "}C & "
						out_excel += str(temp) + " & "
						firstOfRun = False
					elif firstLine:
						out_table += " & \\ang{" + str(temp) + "}C & "
						out_excel += str(temp) + " & "
						firstLine = False
					else:
						out_table += " & & "
						out_excel += " & & "
					out_table += (string + " \\\\ \\hline\n")
					out_excel += (string + "\n")
				if len(listOfResults) > 1:
					R_err = 0
					for thisOne in range(len(listOfResults[1])):
						R_err += listOfResults[1][thisOne]**2
					R_data = str(np.round(listOfResults[1][0], 3)) + "(" + str(np.round(np.sqrt(R_err), 3)) + ")"
					for thisOne in range(len(listOfResults[2])):
						R_err += listOfResults[1][thisOne]**2
					R_op = str(np.round(listOfResults[2][0], 3)) + "(" + str(np.round(np.sqrt(R_err), 3)) + ")"
					allRatios_err += R_err
					allRatios.append(listOfResults[2][0])

					out_table += "\\multicolumn{" + str(6 + low) + "}{|c|}{Alk.Ratio (K:Rb): " + R_data + " \\textrightarrow$\\,$ at \\ang{" + str(235) + "}C: " + R_op + "} \\\\ \\hline\n"
					out_excel += "Alk.Ratio (K:Rb) & " + R_data + " & at 235C & " + R_op + "\n"
				else:
					out_excel += "\n"
			if len(allRatios) > 0:
				final_err = np.sqrt(np.std(allRatios)**2 + allRatios_err)
				final_txt = str(np.round(np.mean(allRatios), 3)) + "(" + str(np.round(final_err, 3)) + ")"
				out_table += "\\multicolumn{" + str(6 + low) + "}{|c|}{\\textrightarrow \\quad \\textrightarrow \\quad Avg. Estimated Alk. Ratio (K:Rb) at \\ang{" + str(235) + "}C: " + final_txt + " \\quad \\textleftarrow \\quad \\textleftarrow} \\\\ \\hline\n"
			out_excel += "Avg. Alk.Ratio (K:Rb) & " + final_txt + "\n"

			if (target in highSets) and (int(runNum) in highSets[target]):
				file = Path(Path.cwd(), "__AllData", "tableData_high.txt")
				writer = open(file, "a")
				writer.write(out_table)
				writer.close()
				file = Path(Path.cwd(), "__AllData", "excelData_high.txt")
				writer = open(file, "a")
				writer.write(out_excel)
				writer.close()
			else:
				file = Path(Path.cwd(), "__AllData", "tableData_low.txt")
				writer = open(file, "a")
				writer.write(out_table)
				writer.close()
				file = Path(Path.cwd(), "__AllData", "excelData_low.txt")
				writer = open(file, "a")
				writer.write(out_excel)
				writer.close()


def rigamaroll(low, high):
	f = [0.34231, 0.69577, 0.334, 0.672] # [RbD1, RbD2, KD1, KD2]
	errorFlag = False
	target = ""
	run, temp = (-1 for _ in range(2))

	val_FWHM, val_cen, val_eta, val_att = ([[],[],[],[]] for _ in range(4))
	err_FWHM, err_cen, err_eta, err_alkRatio_dataTemp, err_alkRatio_opTemp, this_fwhm_err, this_fwhm = ([0, 0, 0, 0] for _ in range(7))
	val_alkRatio_dataTemp, val_alkRatio_opTemp = ([] for _ in range(2))
	err_alkRatio_dataTemp, err_alkRatio_opTemp = (0 for _ in range(2))

	results, dictOfTemps, dictOfRuns = ({} for _ in range(3))

	firstFile = True
	for file in dataPath.glob("*.txt"):

		thisTarget = str(file.name.split("_")[0])
		thisRun = int(re.sub('run', '', file.name.split("_")[1]))
		thisTemp = int(re.sub('C', '', file.name.split("_")[2]))

		if firstFile:
			temp = thisTemp
			run = thisRun
			target = thisTarget
			firstFile = False

		if (thisTarget != target):
			dictOfTemps[temp] = [[slapDash(low, val_FWHM, val_cen, val_att, val_eta, err_FWHM, err_cen, err_eta)]]
			if len(val_alkRatio_dataTemp) > 0:
				dictOfTemps[temp].append([np.mean(val_alkRatio_dataTemp), np.std(val_alkRatio_dataTemp), np.sqrt(err_alkRatio_dataTemp)])
				dictOfTemps[temp].append([np.mean(val_alkRatio_opTemp), np.std(val_alkRatio_opTemp), np.sqrt(err_alkRatio_opTemp)])
			dictOfRuns[run] = dictOfTemps
			results[target] = dictOfRuns

			dictOfTemps, dictOfRuns = ({} for _ in range(2))

			temp = thisTemp
			run = thisRun
			target = thisTarget
			val_FWHM, val_cen, val_eta, val_att = ([[],[],[],[]] for _ in range(4))
			err_FWHM, err_cen, err_eta, err_alkRatio_dataTemp, err_alkRatio_opTemp, this_fwhm_err, this_fwhm = ([0, 0, 0, 0] for _ in range(7))
			val_alkRatio_dataTemp, val_alkRatio_opTemp = ([] for _ in range(2))
			err_alkRatio_dataTemp, err_alkRatio_opTemp = (0 for _ in range(2))
			if errorFlag:
				errorLogger("tar")
				errorFlag = False

		elif (thisRun != run):
			dictOfTemps[temp] = [[slapDash(low, val_FWHM, val_cen, val_att, val_eta, err_FWHM, err_cen, err_eta)]]
			if len(val_alkRatio_dataTemp) > 0:
				dictOfTemps[temp].append([np.mean(val_alkRatio_dataTemp), np.std(val_alkRatio_dataTemp), np.sqrt(err_alkRatio_dataTemp)])
				dictOfTemps[temp].append([np.mean(val_alkRatio_opTemp), np.std(val_alkRatio_opTemp), np.sqrt(err_alkRatio_opTemp)])
			dictOfRuns[run] = dictOfTemps

			dictOfTemps = {}

			temp = thisTemp
			run = thisRun
			val_FWHM, val_cen, val_eta, val_att = ([[],[],[],[]] for _ in range(4))
			err_FWHM, err_cen, err_eta, err_alkRatio_dataTemp, err_alkRatio_opTemp, this_fwhm_err, this_fwhm = ([0, 0, 0, 0] for _ in range(7))
			val_alkRatio_dataTemp, val_alkRatio_opTemp = ([] for _ in range(2))
			err_alkRatio_dataTemp, err_alkRatio_opTemp = (0 for _ in range(2))
			if errorFlag:
				errorLogger("run")
				errorFlag = False

		elif (thisTemp != temp):
			dictOfTemps[temp] = [[slapDash(low, val_FWHM, val_cen, val_att, val_eta, err_FWHM, err_cen, err_eta)]]
			if len(val_alkRatio_dataTemp) > 0:
				dictOfTemps[temp].append([np.mean(val_alkRatio_dataTemp), np.std(val_alkRatio_dataTemp), np.sqrt(err_alkRatio_dataTemp)])
				dictOfTemps[temp].append([np.mean(val_alkRatio_opTemp), np.std(val_alkRatio_opTemp), np.sqrt(err_alkRatio_opTemp)])

			temp = thisTemp
			val_FWHM, val_cen, val_eta, val_att = ([[],[],[],[]] for _ in range(4))
			err_FWHM, err_cen, err_eta, err_alkRatio_dataTemp, err_alkRatio_opTemp, this_fwhm_err, this_fwhm = ([0, 0, 0, 0] for _ in range(7))
			val_alkRatio_dataTemp, val_alkRatio_opTemp = ([] for _ in range(2))
			err_alkRatio_dataTemp, err_alkRatio_opTemp = (0 for _ in range(2))
			if errorFlag:
				errorLogger("tmp")
				errorFlag = False

		for fitSelect in range(low, high):
			thisFile = "\"" + str(file) + "\""
			included = [False, False, False, False]
			os.system(f"python {program} {thisFile} {fitSelect}")
			val_c0, err_c0 = ([0, 0, 0, 0] for _ in range(2))
			try:
				importDict = importAndUnpickle(Path(picklePath, "exportedDict_specAnalysis.pickle"))
				os.remove(Path(picklePath, "exportedDict_specAnalysis.pickle"))
				#textFile.write(file.name + str(importDict) + "\n\n")
				for line_idx, line_val in enumerate(line_nameType):

					if ("chi2" in importDict[line_val]) and not np.isnan( float(importDict[line_val]["chi2"]) ):
						included[line_idx] = True
						val_FWHM[line_idx].append(importDict[line_val]["val_FWHM"])
						val_cen[line_idx].append(importDict[line_val]["val_cen"])
						val_c0[line_idx] = importDict[line_val]["val_c0"]
						val_att[line_idx].append(importDict[line_val]["val_maxAtt"])

						if fitSelect > 0:
							err_FWHM[line_idx] += importDict[line_val]["err_FWHM"]**2
							this_fwhm_err[line_idx] = importDict[line_val]["err_FWHM"]
							err_cen[line_idx] += importDict[line_val]["err_cen"]**2
							err_c0[line_idx] = importDict[line_val]["err_c0"]
							this_fwhm[line_idx] = importDict[line_val]["val_FWHM"]

						if fitSelect == 2:
							val_eta[line_idx].append(importDict[line_val]["val_eta"])
							err_eta[line_idx] += importDict[line_val]["err_eta"]**2

				for i in range(2):
					if included[i] and included[i+2]:
						if fitSelect == 0:
							R = alkDenRatio_integral(val_c0[i], f[i], val_c0[i+2], f[i+2])
						else:
							R = alkDenRatio_fitted(val_c0[i], this_fwhm[i], f[i], val_c0[i+2], this_fwhm[i+2], f[i+2])
							R_err = (R*err_c0[i+2]/val_c0[i+2])**2 + (R*this_fwhm_err[i+2]/this_fwhm[i+2])**2 + (R*err_c0[i]/val_c0[i])**2 + (R*this_fwhm_err[i]/this_fwhm[i])**2
							err_alkRatio_dataTemp += R_err**2
							err_alkRatio_opTemp += (R_err*D(235)/D(temp))**2
						
						val_alkRatio_dataTemp.append( R )
						val_alkRatio_opTemp.append(R*D(235)/D(temp))
						
			except:
				errorLogger(f"{file.name} ({fitSelect}) {line_fitType[fitSelect]}")
				errorFlag = True
		print("\nTime Elapsed: " + str(datetime.now() - timeElapsed[0]).split(".")[0] + "\n")
	dictOfTemps[temp] = [[slapDash(low, val_FWHM, val_cen, val_att, val_eta, err_FWHM, err_cen, err_eta)]]
	if len(val_alkRatio_dataTemp) > 0:
		dictOfTemps[temp].append([np.mean(val_alkRatio_dataTemp), np.std(val_alkRatio_dataTemp), np.sqrt(err_alkRatio_dataTemp)])
		dictOfTemps[temp].append([np.mean(val_alkRatio_opTemp), np.std(val_alkRatio_opTemp), np.sqrt(err_alkRatio_opTemp)])
	dictOfRuns[run] = dictOfTemps
	results[target] = dictOfRuns
	return results


'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOPERATION: DO IT SO MANY FUCKING TIMES\n".center(174, "*"))
print()

pathChecker([listOfPaths])

startStamp = stamp()
program = "spec_cleanerAnalysis.py"

line_nameType = ["RbD1", "RbD2", "KD1", "KD2"]
line_infoType = ["val_c0", "err_c0", "val_FWHM", "err_FWHM", "val_cen", "err_cen", "val_maxAtt", "val_eta", "err_eta"]
line_fitType = ["Intgrated", "Lorentzian", "PseudoVoigt"]



######################################## MEAT AND POTATOES ###################################

pathChecker([picklePath, dataPath, errorPath])
startStamp = stamp()

results = rigamaroll(1, 3)
cut_print(results, 1)

results = rigamaroll(0, 1)
cut_print(results, 0)


if False:
	for target, dictOfRuns in results.items():
		print(target)
		for runNum, dictOfTemps in dictOfRuns.items():
			print(f"\tRun {runNum}")
			for temp, listOfResults in dictOfTemps.items():
				print(f"\t\t{temp}")
				for line, string in listOfResults[0][0].items():
					print(f"\t\t\t{line}\n\t\t\t\t{string}")

				for i in range(1, len(listOfResults)):
					print(f"\t\t\t{listOfResults[i]}")

endStamp = stamp()
print(f"\n****** Ran from {startStamp} to {endStamp} ******\n")
diff = str(timeElapsed[len(timeElapsed)-1] - timeElapsed[0]).split(".")[0]
print(f"\t\t --> Total time elapsed: {diff}\n\n")
			
print()
print("\nOPERATION: SAT\n".center(150, "*"))
print()