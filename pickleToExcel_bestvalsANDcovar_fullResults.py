'''----------------------------------------Imports----------------------------------------
'''
import sys, os, re, pickle, copy, collections
import numpy as np
from pathlib import Path
import scipy.stats as stats
from datetime import datetime
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

#  makeMyDict(data_highPow, targetName, fitIDX, runNum, temperature)
def makeMyDict(dic, tar, run, tmp):
	if tar not in dic:
		dic[tar] = {}
	if run not in dic[tar]:
		dic[tar][run] = {}
	if tmp not in dic[tar][run]:
		dic[tar][run][tmp] = [[(0,0), (0,0), (0,0)] for _ in range(4)]
	return dic

def stamp():
	x = datetime.now()
	startDate = re.sub('-', '', str(x.date()))
	startTime = str(x.time()).split(".")[0]
	startTime = startTime.split(":")[0] + "h" + startTime.split(":")[1] + "m" + startTime.split(":")[2] + "s"
	return f"D{startDate}_T{startTime}"

def extendo(row):
	for _ in range(len(row), 7):
		row.append("")
	return row

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





### thisDict[targetName][runNum][temp][lineNumber][fitNumber]["best_vals"] = [[set1_best_vals], [set2_best_vals], ......]
			

'''

if updates:
	print()
	print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
	print()

allTargetDict = import_unpickle(Path(Path.cwd(), "__AllData", "exported_franken.pickle"))

defs_fitNames = ["Integral", "Lorentzian", "Pseudo-Voigt"]
defs_lineNames = ["RbD1", "RbD2", "KD1", "KD2"]
labels_highPow = [("Brianna", [1]), ("Florence", [1, 2]), ("Fulla", [1, 2, 3, 4, 5]), ("Kappa1", [2]), ("Kappa2", [1]), ("Kappa3", [1]), ("Kappa4", [1, 2]), ("Noah", [1]), ("SandyII", [1, 2]), ("Savior", [1]), ("Wayne", [1])]
labels_lowPow = [("Austin", [1]), ("BigBrother", [1]), ("Brianna", [2]), ("Butterball", [1, 2]), ("Dutch", [1]), ("Florence", [3, 4, 5]), ("Fulla", [6]), ("Kappa2", [2, 3]), ("Kappa3", [2, 3, 4]), ("Kappa4", [3, 4]), ("Kappa5", [1]), ("Noah", [2]), ("Tommy", [1]), ("Wayne", [2])]

bestVals_int = ["integral", "peak_y", "peak_x", "FWHM"]
bestVals_lor = ["c0", "c1", "c2", "c3", "m", "b"]
bestVals_pVt = ["c0", "c1", "c2", "c3", "eta", "m", "b"]

bestVals_name = ["val_c0", "err_c0", "val_FWHM", "err_FWHM", "val_cen", "err_cen", "val_maxAtt", "val_eta", "err_eta", "chi2"] # [bestVals_int, bestVals_lor, bestVals_pVt]

densitys = {"Austin":[7.497997593, 0.044300479, 0.114377003, 0.001058141], "BigBrother":[6.992, 0.042338519, 0.109213049, 0.001034741], "Brianna":[6.467520579, 0.057384012, 0.109645416, 0.001363965], "Butterball":[7.608155424, 0.045385463, 0.113826602, 0.001068162], "Dutch":[7.758636401, 0.044300479, 0.115514796, 0.001101919], "Florence":[6.920131167, 0.061364371, 0.111956571, 0.001352082], "Fulla":[6.736854665, 0.058986278, 0.119623728, 0.001445024], "Kappa1":[0.777, 0.026, 0.092, 0.003], "Kappa2":[0.775, 0.027, 0.092, 0.003], "Kappa3":[0.803, 0.028, 0.092, 0.002], "Kappa4":[0.788, 0.023, 0.095, 0.003], "Kappa5":[0.794, 0.025, 0.091, 0.003], "Noah":[7.062238705, 0.063027581, 0.113721702, 0.001431226], "ProtovecII":[0, 0, 0, 0], "SandyII":[7.46, 0.1, 0.109797738, 0.005], "Savior":[7.263426419, 0.064400946, 0.111089111, 0.001377022], "Tommy":[7.76, 0.1, 0.109797738, 0.005], "Wayne":[7.312580238, 0.043481186, 0.114343142, 0.001071954]} ## [3He, 3He_err, N2, N2_err]

tempRow_kappa = ["", "Temp."] + list(range(50, 135, 5))
tempRow_target = ["", "Temp."] + list(range(110, 170, 5))


for powIDX, powLIST in enumerate([labels_lowPow, labels_highPow]):
	exportData = {}
	for tarIDX, tarTUPLE in enumerate(powLIST):
		tarName = tarTUPLE[0]
		runList = tarTUPLE[1]
		for runIDX, runNum in enumerate(runList):
			for temperature, listOfLines in allTargetDict[tarName][runNum].items():
				exportData = copy.deepcopy(makeMyDict(exportData, tarName, runNum, temperature))
				for line in range(4):
					if len(listOfLines[line][0]["chi2"]) > 0:
						for fit in range(0, 3):
							exportBestVals = []
							exportBestErrs = []
							val, err, counter = (0 for _ in range(3))

							### CODE BROKEN




							for val_idx in range(len(listOfLines[line][fit]["best_vals"][0])):
								
								



								for dataSet_idx in range(len(listOfLines[line][fit]["best_vals"])):
									val += listOfLines[line][fit]["best_vals"][dataSet_idx][val_idx]
									if fit != 0:
										err += listOfLines[line][fit]["covar"][dataSet_idx][val_idx]**2
									counter += 1
								


								exportBestVals.append(val / counter)
								exportBestErrs.append(np.sqrt(err) / counter)
							exportData[tarName][runNum][temperature][line][fit] = (exportBestVals, exportBestErrs)







	fileName = "Results_" + str(["low", "high"][powIDX]) + "Power_" + stamp() + ".xlsx"
	out = Path(Path.cwd(), "__AllData", fileName)

	with pd.ExcelWriter(out) as writer:
		for tarName, runDict in exportData.items():
			output = []
			for fitIdx in range(3):
				if fitIdx == 0:
					row = ["3He", densitys[tarName][0], densitys[tarName][1]]
					output.append(extendo(row))
					row = ["N2", densitys[tarName][2], densitys[tarName][3]]
					output.append(extendo(row))
				row = [defs_fitNames[fitIdx], "", ""]
				for bestIdx, bestVal in enumerate(bestVals_name[fitIdx]):
					row.append(bestVal)
				output.append(extendo(row))
				
				for runNum, tempDict in runDict.items():
					flag1st_run = True
					for temp, lineList in tempDict.items():
						flag1st_line = True
						for lineIdx, fitList in enumerate(lineList):
							if flag1st_run:
								row = [f"Run {runNum}"]
								flag1st_run = False
							else:
								row = [""]
							if flag1st_line:
								row.append(temp)
								flag1st_line = False
							else:
								row.append("")
							row.append(defs_lineNames[lineIdx])
							err = ["", "", "err"]

							for idx in range(len(bestVals_name[fitIdx])):
								try:
									row.append(fitList[fitIdx][0][idx])
									err.append(fitList[fitIdx][1][idx])
								except:
									pass
							output.append(extendo(row))
							output.append(extendo(err))
				output.append(extendo([]))
				output.append(extendo([]))
				

			df = pd.DataFrame(output)
			df.to_excel(writer, header=None, index=None, sheet_name=f"{tarName}")


if updates:
	print()
	print("\nSee you, Space Cowboy...\n".center(150, "-"))
	print()
