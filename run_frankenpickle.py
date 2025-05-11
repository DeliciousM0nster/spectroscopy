'''----------------------------------------Imports----------------------------------------
'''

import sys, os, re, glob, pickle, collections
import numpy as np
import math as m
from datetime import datetime
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
	path = Path(".", "__AllData", fileIn)
	with open(path, 'rb') as x:
		temp = pickle.load(x)
	return temp

### Notes "Dunk it in the brine!"
	# Purpose: Export a serialized file (read by "run_multiSpec.py")
	# Takes: n (thing being pickled), line (RbD1, RbD2, KD1D2)
	# Exports: Pickled file (usually of residuals) labelled with the line it's for
def export_toPickle(pPath, n, f):
	path = Path(pPath, f)
	with open(path, 'wb') as e:
		pickle.dump(n, e)
	return str(path)

def stamp():
	startDate = re.sub('-', '', str(datetime.now().date()))
	startTime = str(datetime.now().time()).split(".")[0]
	startTime = startTime.split(":")[0] + "h" + startTime.split(":")[1] + "m" + startTime.split(":")[2] + "s"
	return f"D{startDate}_T{startTime}"

'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
print()

picklePath = Path(Path.cwd(), "__AllData", "__donePickles")
exportPath = Path(Path.cwd(), "__AllData")

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

'''


exportDict = {}


for file in picklePath.glob("*.pickle"):
	pickleDict = importAndUnpickle(file)
	for targetName, listOfRuns in pickleDict.items():
		if targetName not in exportDict:
			exportDict[targetName] = [{} for _ in range(10)]
		for runNumber, dictOfTemps in enumerate(listOfRuns):
			for temp, listOfLines in dictOfTemps.items():
				exportDict[targetName][runNumber][temp] = listOfLines
			exportDict[targetName][runNumber] = collections.OrderedDict(sorted(exportDict[targetName][runNumber].items()))

exportDict = collections.OrderedDict(sorted(exportDict.items()))

for tar_name, tar_listOfRuns in exportDict.items():
	print(f"\n{tar_name}")
	for run_idx, run_temps in enumerate(tar_listOfRuns):
		if run_temps:
			print(f"\tRun number {run_idx}")
			for temp_val, temp_lineData in run_temps.items():
				print(f"\t\tTemp: {temp_val}")

export_toPickle(exportPath, exportDict, "exported_franken.pickle")

print()
print("\nSee you, Space Cowboy...\n".center(150, "-"))
print()

