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
import collections
from pathlib import Path
degSymbol = u"\u00b0"

'''----------------------------------------Flags and Options----------------------------------------
'''


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
	x = str(message) + ".pickle"
	path = Path(pPath, x)
	with open(path, 'wb') as e:
		pickle.dump(n, e)
	return str(path)

def stamp():
	startDate = re.sub('-', '', str(datetime.now().date()))
	startTime = str(datetime.now().time()).split(".")[0]
	startTime = startTime.split(":")[0] + "h" + startTime.split(":")[1] + "m" + startTime.split(":")[2] + "s"
	return f"D{startDate}_T{startTime}"

def exportToTextFile(path, lowORhigh, fitORint, tar, text):
	fullPath = Path(path, lowORhigh, fitORint)
	f = Path(fullPath, f"{tar}.txt")
	pathChecker([fullPath])
	writer = open(f, "a")
	writer.write(text)
	writer.close()

def debugger(name, val):
	if troubleshooting:
		print(f"\t---> {name} = {val}")

def alkDenRatio_integral(I_d1, f_d1, I_d2, f_d2):
	return (I_d2/I_d1)*(f_d1/f_d2)

def alkDenRatio_fitted(c_d1, g_d1, f_d1, c_d2, g_d2, f_d2):
	return (c_d2/c_d1)*(f_d1/f_d2)*(g_d1/g_d2)

def pathChecker(pList):
	for i, p in enumerate(pList):
		try:
			os.makedirs(p)
		except:
			pass

'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
print()



troubleshooting = False

for lh_idx, lowORhigh in enumerate(["low", "high"]):
	for f_idx, fitORint in enumerate(["integral", "fitted"]):

		firstTemp = False
		picklePath = Path(Path.cwd(), "__AllData")
		pathChecker([picklePath])
		allTargetInfo = importAndUnpickle(Path(picklePath, f"{lowORhigh}Temp{fitORint.upper()}_clean.pickle"))

		targetDataKeys = ["FWHM", "center", "maxAtt"]
		if fitORint == "fitted":
			targetDataKeys.append("eta")

		lineNamesByIndex = ["RbD1", "RbD2", "KD1", "KD2"]
		oscStrengths = [0.34231, 0.69577, 0.334, 0.672]

		#temporaryDict = { "FWHM":[], "center":[], "c0":[], "integral":[], "chi2":[], "eta":[], "maxAtt":[], "fitUsed":[] }

		for tar_name, listOfRuns in allTargetInfo.items():
			print(tar_name)
			
			outputString = ""
			if fitORint == "fitted":
				outputString += " \t \t \tFitted"
			else:
				outputString += " \t \t \tIntegral"
			outputString += "\n \t \t \tFWHM (GHz)\tCenter (GHz)\tMax.Att."
			if fitORint == "fitted":
				outputString += "\teta"
			outputString += "\n"

			for runNumber, dictOftemps in enumerate(listOfRuns):
				if bool(dictOftemps):
					outputString += f"Run {runNumber}\t"
					firstTemp = True
					debugger
					for temp, listOfLines in dictOftemps.items():
						included = [False for _ in range(4)]
						if not firstTemp:
							outputString += " \t"
						else:
							firstTemp = False
						outputString += f"{temp}{degSymbol}C\t"
						for lineIdx, lineDataDict in enumerate(listOfLines):
							if lineIdx != 0:
								outputString += " \t \t"
							if lineDataDict["fitUsed"]:
								included[lineIdx] = True
								outputString += f"{lineNamesByIndex[lineIdx]}\t"
								for dataIdx, dataKey in enumerate(targetDataKeys):
									if dataKey == "eta":
										entry = np.mean([i for i in lineDataDict[dataKey] if i != 0])
									else:
										entry = np.mean(lineDataDict[dataKey])
									outputString += f"{entry}\t"
							outputString += "\n"

						alkDenRatio = []
						for K in range(2,4):
							for Rb in range(0,2):
								if included[Rb] and included[K] and (abs(K-Rb) == 2):
									if troubleshooting: outputString += f" \t[{lineNamesByIndex[K]}:{lineNamesByIndex[Rb]}]\t"
									if fitORint == "fitted":
										rat = alkDenRatio_fitted(np.mean(listOfLines[Rb]["c0"]), np.mean(listOfLines[Rb]["FWHM"]), oscStrengths[Rb], np.mean(listOfLines[K]["c0"]), np.mean(listOfLines[K]["FWHM"]), oscStrengths[K])
										alkDenRatio.append(rat)
										if troubleshooting: outputString += str(rat)
									else:
										rat = alkDenRatio_integral(np.mean(listOfLines[Rb]["integral"]), oscStrengths[Rb], np.mean(listOfLines[K]["integral"]), oscStrengths[K])
										alkDenRatio.append(rat)
										if troubleshooting: outputString += str(rat)
									if troubleshooting: outputString += "\n"
						if alkDenRatio:
							entry = np.mean(alkDenRatio)
							outputString += f"\t\tAlk.Den.Ratio\t{entry}\t"
						outputString += "\n\n"
			exportToTextFile(picklePath, lowORhigh, fitORint, tar_name, outputString)







### allTargetInfo["targetName"][runNumber][temp][indForLine] = { "FWHM":[], "center":[], "c0":[], "integral":[], "chi2":[], "eta":[], "maxAtt":[], "fitUsed":[] }



print()
print("\nSee you, Space Cowboy...\n".center(150, "-"))
print()


### Low: just got rid of kappa3 run3 70. others were gone