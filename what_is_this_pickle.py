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
def importAndUnpickle():
	temp = []
	path = Path(".", "__AllData", sys.argv[1])
	with open(path, 'rb') as x:
		temp = pickle.load(x)
	return temp

'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
print()

lineNames = ["RbD1", "RbD2", "KD1", "KD2"]
line_fitType = ["Integral", "Lorentzian", "pseudoVoigt"]

theThing = importAndUnpickle()


#print(theThing)


#for k, v in theThing.items():
#	print(f"{k}: {v}")


for cellName, runList in theThing.items():
	print(cellName)
	for runNumber, tempDict in enumerate(runList):
		print(f"\t{runNumber}")
		for temp, lineList in tempDict.items():
			print(f"\t\t{temp}")
			for fitIdx, fitList in enumerate(lineList):
				print(f"\t\t\t{lineNames[fitIdx]}")
				for dataSet_idx, dataSet_dataDict in enumerate(fitList):
					print(f"\t\t\t\t{line_fitType[dataSet_idx]}")
					for dataName, data in dataSet_dataDict.items():
						print(f"\t\t\t\t\t{dataName}: {(data)}")




### thisDict[targetName][runNum][temp][lineNumber][fitNumber][dataName] = [[dataSet1], [dataSet2], ......]


print()
print("\nSee you, Space Cowboy...\n".center(150, "-"))
print()
