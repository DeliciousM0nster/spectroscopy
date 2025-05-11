'''----------------------------------------Imports----------------------------------------
'''

import sys
import os
import glob
import pickle
import numpy as np
import math as m
import re

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

'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOPERATION: DO IT SO MANY FUCKING TIMES\n".center(174, "*"))
print()

pathChecker([listOfPaths])

startStamp = stamp()
program = "spec_cleanerAnalysis.py"

line_nameType = ["RbD1", "RbD2", "KD1", "KD2"]
#line_infoType = ["best_vals", "covar", "chi2"]
line_infoType = ["val_c0", "err_c0", "val_FWHM", "err_FWHM", "val_cen", "err_cen", "val_maxAtt", "val_eta", "err_eta", "chi2"]
line_fitType = ["Integral", "Lorentzian", "pseudoVoigt"]

target = ""
run, temp = (-1 for _ in range(2))

listOfRuns = [{} for _ in range(10)]
dictOfTemps = {}
#aFreshDictionary = {"best_vals":[], "covar":[], "chi2":[]}
aFreshDictionary = {"val_c0":[], "err_c0":[], "val_FWHM":[], "err_FWHM":[], "val_cen":[], "err_cen":[], "val_maxAtt":[], "val_eta":[], "err_eta":[], "chi2":[]}
listOfLines = []
for _ in range(4):
	listOfFits = [ copy.deepcopy(aFreshDictionary) for _ in range(3) ]
	listOfLines.append(listOfFits)


'''
input:

best_vals, covar, chi2

"val_c0":c0, "err_c0":c0_std, "val_FWHM":FWHM, "err_FWHM":FWHM_std, "val_cen":c2, "err_cen":c2_std, "val_maxAtt":maxAtt, "val_eta":eta, "err_eta":eta_std, "x":x, "y":y, "yFit":yFit, "chi2":chi2

output:

TargetDict{
	"TargetName":[
		{temps_run0}, 
		{"temp_in_run1":[
			RbD1_fits, 
			RbD2_fits, [
				fit0, 
				fit1, 
				{"dataType":[listOfData]}], 
			KD2_fits]}, 
		{temps_run3}], 
	"nextTarget":[]}
'''

######################################## MEAT AND POTATOES ###################################

pathChecker([picklePath, dataPath, errorPath])
targetDict = {}

errorFlag = False

for file in dataPath.glob("*.txt"):

	thisTarget = str(file.name.split("_")[0])
	thisRun = int(re.sub('run', '', file.name.split("_")[1]))
	thisTemp = int(re.sub('C', '', file.name.split("_")[2]))

	if (thisTarget != target):
		if target != "":
			dictOfTemps[temp] = []
			for line_idx, line_dict in enumerate(listOfLines):
				dictOfTemps[temp].append(copy.deepcopy(line_dict))
			listOfRuns[run] = copy.deepcopy(dictOfTemps)
			targetDict[target] = listOfRuns
			del dictOfTemps
		listOfRuns = [{} for _ in range(10)]
		dictOfTemps = {}
		listOfLines = []
		for _ in range(4):
			listOfFits = [ copy.deepcopy(aFreshDictionary) for _ in range(3) ]
			listOfLines.append(listOfFits)
		target = thisTarget
		run = thisRun
		temp = thisTemp
		if errorFlag:
			errorLogger("tar")
			errorFlag = False
	elif (thisRun != run):
		dictOfTemps[temp] = []
		for line_idx, line_dict in enumerate(listOfLines):
			dictOfTemps[temp].append(copy.deepcopy(line_dict))
		listOfRuns[run] = copy.deepcopy(dictOfTemps)
		del dictOfTemps
		dictOfTemps = {}
		listOfLines = []
		for _ in range(4):
			listOfFits = [ copy.deepcopy(aFreshDictionary) for _ in range(3) ]
			listOfLines.append(listOfFits)
		run = thisRun
		temp = thisTemp
		if errorFlag:
			errorLogger("run")
			errorFlag = False
	elif (thisTemp != temp):
		dictOfTemps[temp] = []
		for line_idx, line_dict in enumerate(listOfLines):
			dictOfTemps[temp].append(copy.deepcopy(line_dict))
		listOfLines = []
		for _ in range(4):
			listOfFits = [ copy.deepcopy(aFreshDictionary) for _ in range(3) ]
			listOfLines.append(listOfFits)
		temp = thisTemp
		if errorFlag:
			errorLogger("tmp")
			errorFlag = False

	for fitSelect in range(0,3):
		thisFile = "\"" + str(file) + "\""
		os.system(f"python {program} {thisFile} {fitSelect}")
		try:
			importDict = importAndUnpickle(Path(picklePath, "exportedDict_specAnalysis.pickle"))
			os.remove(Path(picklePath, "exportedDict_specAnalysis.pickle"))
			for line_idx, line_val in enumerate(line_nameType):
				if (line_val in importDict) and not np.isnan( float(importDict[line_val]["chi2"] )):
					for info_idx, info_val in enumerate(line_infoType):
						if info_val in importDict[line_val]:
							listOfLines[line_idx][fitSelect][info_val].append(importDict[line_val][info_val])
		except:
			errorLogger(f"{file.name} ({fitSelect}) {line_fitType[fitSelect]}")
			errorFlag = True

	print("\nTime Elapsed: " + str(datetime.now() - timeElapsed[0]).split(".")[0] + "\n")

dictOfTemps[temp] = []
for line_idx, line_dict in enumerate(listOfLines):
	dictOfTemps[temp].append(copy.deepcopy(line_dict))
listOfRuns[run] = copy.deepcopy(dictOfTemps)
targetDict[target] = listOfRuns

endStamp = stamp()

m = f"_{endStamp}"
x = export_toPickle(picklePath, targetDict, m)
print(f"All target data exported here:\t{x}")

endStamp = stamp()
print(f"\n****** Ran from {startStamp} to {endStamp} ******\n")
diff = str(timeElapsed[len(timeElapsed)-1] - timeElapsed[0]).split(".")[0]
print(f"\t\t --> Total time elapsed: {diff}\n\n")

print()
print("\nOPERATION: SAT\n".center(150, "*"))
print()