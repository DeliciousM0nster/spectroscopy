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

lowORhigh = "low"

targetedForDeletion = [("Dutch", 1, 130)] ### [("target", "run", "temp")]

fileName = str(sys.argv[1]).split(".")[0]

picklePath = Path(Path.cwd(), "__AllData")
allTargetInfo = importAndUnpickle(Path(picklePath, f"{fileName}.pickle"))

for targetIdx, targetInfo in enumerate(targetedForDeletion):
	targetName, runNumber, temperature = targetInfo
	for tar_name, tar_listOfRuns in allTargetInfo.items():
		if tar_name == targetName:
			print(f"\n{tar_name}")
			for run_idx, run_temps in enumerate(tar_listOfRuns):
				print(f"\tRun number {run_idx}")
				for temp_val, temp_lineData in run_temps.items():
					if run_idx == runNumber:
						print(f"\t\tTemp: {temp_val}")

	print(f"\nSnip out {targetName} run{runNumber} {temperature}C? (y/n)")
	answer = input()
	if answer.lower() == "y":
		try:
			del allTargetInfo[targetName][runNumber][temperature]
			print(f"\nSnipped! ({targetName} run{runNumber} {temperature}C is TOAST!)")
		except:
			print(f"\nIt doesn't seem to exist!")
	else:
		print(f"\nExit Program? (y/n)")
		answer = input()
		if answer.lower() == "y":
			sys.exit()

	for tar_name, tar_listOfRuns in allTargetInfo.items():
		if tar_name == targetName:
			print(f"\n{tar_name}")
			for run_idx, run_temps in enumerate(tar_listOfRuns):
				print(f"\tRun number {run_idx}")
				for temp_val, temp_lineData in run_temps.items():
					if run_idx == runNumber:
						print(f"\t\tTemp: {temp_val}")

export_toPickle(picklePath, allTargetInfo, f"{fileName}_clean.pickle")

print()
print("\nSee you, Space Cowboy...\n".center(150, "-"))
print()


### Low: just got rid of kappa3 run3 70. others were gone