# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:08:04 2020

@author: ahinoamp

Plot prior and posterior histograms of parameter values
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import FormatStrFormatter

# I've explicitly declared my path as being in Windows format, so I can use forward slashes in it.


DataTypes = ['Grav','Mag','Tracer','GT','FaultMarkers']
DataTypesLabels = ['Gravity','Magnetics','Tracer','Granite Top','Wellbore Fault']

folderresults = os.path.abspath(os.path.join(os.path.dirname( __file__), '..', 'Combo_Scratch'))
file=folderresults+'/'+'ParametersWithMismatch.csv'
Data = pd.read_csv(file)

Data['Status'] = 'Prior'
PercentPosterior = 5
threshold = np.percentile(Data['MinMismatch'], 5)
filterPosterior = Data['MinMismatch'] < 0.51
print('number of prior: ' + str(len(Data)))
print('number of posterior: ' + str(np.sum(filterPosterior)))
Data.loc[filterPosterior, 'Status'] = 'Posterior'    
filterV = Data['Status'] == 'Posterior'  
minMismatchList = Data['MinMismatch']

opacity = 0.6
blue = [95/255.0,221/255.0,229/255.0, opacity]
red = [217/255.0,32/255.0,39/255.0, opacity]
nbins=35
ytop = 800
ytoppost = 40

fig, axs = plt.subplots(1, 6, figsize=(15,2.5))
ax = axs[0]
n, bins, patches = ax.hist(minMismatchList, bins=30, label='Prior',color=blue)

ax2 = ax.twinx()
ax2.hist(minMismatchList[filterV], bins=bins, label='Posterior', color=red)
ax2.set_yticks([])
ax2.set_ylim([0, ytoppost])

ax.set_xlim([0, 1.5])
ax.set_xlabel('Error')
ax.set_ylabel('Count prior')
ax.set_title('(a) Total Error')
ax.set_ylim([0, ytop])
ax.legend()
prefix = ['(b) ', '(c) ', '(d) ', '(e) ', '(f) ']
i=0
for dt in DataTypes:
    print(i)
    ax = axs[i+1]
    if(dt == 'Tracer'):
        n, bins, patches = ax.hist(Data[dt], bins=17, label='Prior',color=blue)
    else:
        n, bins, patches = ax.hist(Data[dt], bins=nbins, label='Prior',color=blue)
    
    ax2 = ax.twinx()
    ax2.hist(Data[dt][filterV], bins=bins, label='Posterior', color=red)
    ax2.set_ylim([0, ytoppost])
    if(i==4):
        ax2.set_ylabel('Count posterior', rotation=270,labelpad=14)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    else:
        ax2.set_yticks([])
        
    ax.set_xlabel('Error')
    ax.set_title(prefix[i] + DataTypesLabels[i] + ' Error')
    ax.set_yticks([])
    ax.set_ylim([0, ytop])
    ax.set_xlim([0, 1.5])
    i=i+1

plt.savefig('C:/Users/ahino/OneDrive - Leland Stanford Junior University/School/Research/PhDResearch/PatuaCollaboration/ErrorHistograms.png', dpi=300,bbox_inches='tight')    
