
'''----------------------------------------Imports----------------------------------------
'''

import sys
import re
import numpy as np
from array import array
#from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import curve_fit
import pandas as pd
import random as rd
import math


'''----------------------------------------Flags and Options----------------------------------------
'''

#### Line Visual Options

mark = '.'
lineStyle = ''
markSize = 1
color = 'r'

#### Figure Labels
xLab = "Freq (GHz)"
yLab = "ln(prim:ref)"

fileTitle = "Test"

allowedError = 1

'''----------------------------------------Functions----------------------------------------
'''


'''----------------------------------------Main----------------------------------------
'''
#print("\n-------------------------------------------------------\nOkay. 3, 2, 1, let's jam!\n-------------------------------------------------------\n")

data = np.loadtxt( sys.argv[1] )
file = str(sys.argv[1]).split("\\")[len(sys.argv[1].split("\\"))-1].split("/")[len(sys.argv[1].split("/")) - 1].split(".")[0]
#cellName = file.split("_")[0]
#temp = re.sub('C', '', file.split("_")[2])

### Split data into signal, wlm, errors, etc
ref = data[ :,1 ]
pri = data[ :,2 ]
wlm = data[ :,3 ]
ratio = np.log(pri/ref)
wlm_clean = []
ratio_clean = []

### Sort by line, remove bad data points (large error)
error = np.abs(data[ :,5 ])
for idx, val in enumerate(ratio):
	if (abs(error[idx]/val) < allowedError):
		wlm_clean.append(wlm[idx])
		ratio_clean.append(val)

del ref
del pri

xData = np.asarray(wlm_clean)
yData = np.asarray(ratio_clean)

plt.figure(file)
plt.plot( xData, yData, color=color, marker=mark, markersize=markSize, ls=lineStyle )
plt.show()

#print("\n-------------------------------------------------------\nSee you, Space Cowboy...\n-------------------------------------------------------\n")


### How to go from covarince matrix to t and p values OR how to calculate t and p values from....optimize fit? least squares?