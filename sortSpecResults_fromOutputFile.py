'''----------------------------------------Imports----------------------------------------
'''
import sys
import re
import numpy as np
from pathlib import Path


'''----------------------------------------Flags and Options----------------------------------------
'''


'''----------------------------------------Functions----------------------------------------
'''

def cleanItUp(s):
	return float(s.split(": ")[2].split("\n")[0])

'''----------------------------------------Main----------------------------------------
'''
print()
print("\nOkay. 3, 2, 1, let's jam! \n".center(152, "-"))
print()

outputRecord = Path(".", "__AllData", "low", "logs", "outputRecord.txt")

info_kappa = {"c0":[], "c1":[], "c3":[], "eta":[], "m":[], "b":[]}
info_other = {"c0":[], "c1":[], "c3":[], "eta":[], "m":[], "b":[]}

kappaFlag = False
i = 1
with open(outputRecord) as file:

	for line in file:
		
		if ("---> " in line.lower()):
			if "kappa" in line.split("--> ")[1].split("_")[0].lower():
				kappaFlag = True
				i += 1
			else:
				kappaFlag = False
				i += 1
		elif (("key" in line) and (float(line.split(": ")[2]) != 0)):
			cleanVal = cleanItUp(line)
			if ("_c0" in line.lower()):
				if kappaFlag:
					info_kappa["c0"].append(cleanVal)
				else:
					info_other["c0"].append(cleanVal)
			elif ("_c1" in line.lower()):
				if kappaFlag:
					info_kappa["c1"].append(cleanVal)
				else:
					info_other["c1"].append(cleanVal)
			elif ("_fwhm" in line.lower()):
				if kappaFlag:
					info_kappa["c3"].append(cleanVal)
				else:
					info_other["c3"].append(cleanVal)
			elif ("_eta" in line.lower()):
				if kappaFlag:
					info_kappa["eta"].append(cleanVal)
				else:
					info_other["eta"].append(cleanVal)
			elif ("_m\t" in line.lower()):
				if kappaFlag:
					info_kappa["m"].append(cleanVal)
				else:
					info_other["m"].append(cleanVal)
			elif ("_b\t" in line.lower()):
				if kappaFlag:
					info_kappa["b"].append(cleanVal)
				else:
					info_other["b"].append(cleanVal)


gran = 7
print(f"\nFor Kappa cells...")
for k, v in info_kappa.items():
	mean = np.round(np.mean(v), gran)
	median = np.round(np.median(v), gran)
	std = np.round(np.std(v), gran)
	minm = np.round(np.min(v), gran)
	maxm = np.round(np.max(v), gran)
	print(f"{k}: {mean} +/- {std}\n\tmin: {minm}\n\tMedian: {median}\n\tmax: {maxm}")

print(f"\nFor other cells...")
for k, v in info_other.items():
	mean = np.round(np.mean(v), gran)
	median = np.round(np.median(v), gran)
	std = np.round(np.std(v), gran)
	minm = np.round(np.min(v), gran)
	maxm = np.round(np.max(v), gran)
	print(f"{k}: {mean} +/- {std}\n\tmin: {minm}\n\tMedian: {median}\n\tmax: {maxm}")


print()
print("\nSee you, Space Cowboy...\n".center(150, "-"))
print()