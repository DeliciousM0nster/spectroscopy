'''----------------------------------------Imports----------------------------------------
'''
import sys, os, re, pickle, copy, collections
import numpy as np
from pathlib import Path
import scipy.stats as stats
import datetime as ddtt
import pandas as pd

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


'''----------------------------------------Main----------------------------------------
'''

'''
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
		{temps_run2}], 
	"nextTarget":[]}

["maxAtt", "c0", "c1", "c2", "c3", "eta", "c0_std", "c1_std", "c2_std", "c3_std", "eta_std", "m", "b", "integral", "chi2", "pVal"]

Integral: 	["maxAtt", "c2", "c3", "integral"]
Fitted: 	["maxAtt", "c0", "c1", "c2", "c3", "eta", "c0_std", "c1_std", "c2_std", "c3_std", "eta_std", "m", "b", "chi2", "pVal"]


			

'''

if updates:
	print()
	print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
	print()

allTargetDict = import_unpickle(Path(Path.cwd(), "__AllData", "exported_franken.pickle"))

defs_colHeaders_lorz = ["maxAtt", "c0", "c1", "c2", "c3", "c0_std", "c1_std", "c2_std", "c3_std", "m", "b", "chi2", "pVal"]
defs_colHeaders_psVt = ["maxAtt", "c0", "c1", "c2", "c3", "eta", "c0_std", "c1_std", "c2_std", "c3_std", "eta_std", "m", "b", "chi2", "pVal"]
defs_colHeaders_intl = ["maxAtt", "c2", "c3", "integral"]
defs_lineNames = ["RbD1", "RbD2", "KD1", "KD2"]

allRows = []
thisRow = []
blankRow = [""]*32

output = Path(Path.cwd(), "__AllData", "output.xlsx")
with pd.ExcelWriter(output) as writer:

	for targetName, listOfRuns in allTargetDict.items():
		allRows = [["", "", "", "Lorentzian", "", "", "", "", "", "", "", "", "", "", "", "", "Psuedo-Voigt" "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Integral" "", "", ""]]
		allRows.append(["", "", ""] + defs_colHeaders_lorz + defs_colHeaders_psVt + defs_colHeaders_intl)
		for runNum, dictOfTemps in enumerate(listOfRuns):
			firstLineFlag_run = True
			for temp, listOfLines in collections.OrderedDict(sorted(dictOfTemps.items())).items():
				firstLineFlag_temp = True
				for lineIdx, listOfFits in enumerate(listOfLines):
					if firstLineFlag_run:
						thisRow = [f"Run {runNum}", temp]
						firstLineFlag_run = False
						firstLineFlag_temp = False
					elif firstLineFlag_temp:
						thisRow = [f"", temp]
						firstLineFlag_temp = False
					else:
						thisRow = [f"", f""]
					thisRow.append(defs_lineNames[lineIdx])
					for hdr_idx, dataType in enumerate(defs_colHeaders_lorz):
						thisDataList = allTargetDict[targetName][runNum][temp][lineIdx][1][dataType]
						if len(thisDataList) == 0:
							thisRow.append("")
						elif len(thisDataList) == 1:
							thisRow.append(float(thisDataList[0]))
						else:
							dataMean = np.mean(thisDataList)
							thisRow.append(float(dataMean))
					for hdr_idx, dataType in enumerate(defs_colHeaders_psVt):
						thisDataList = allTargetDict[targetName][runNum][temp][lineIdx][2][dataType]
						if len(thisDataList) == 0:
							thisRow.append("")
						elif len(thisDataList) == 1:
							thisRow.append(float(thisDataList[0]))
						else:
							dataMean = np.mean(thisDataList)
							thisRow.append(float(dataMean))
					for hdr_idx, dataType in enumerate(defs_colHeaders_intl):
						thisDataList = allTargetDict[targetName][runNum][temp][lineIdx][0][dataType]
						if len(thisDataList) == 0:
							thisRow.append("")
						elif len(thisDataList) == 1:
							thisRow.append(float(thisDataList[0]))
						else:
							dataMean = np.mean(thisDataList)
							thisRow.append(float(dataMean))
					allRows.append(thisRow)
			if runNum != 0:
				allRows.append(blankRow)
		df = pd.DataFrame(allRows)
		df.to_excel(writer, header=None, index=None, sheet_name=targetName)


if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()
