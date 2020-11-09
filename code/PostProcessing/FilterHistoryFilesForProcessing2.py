# -*- coding: utf-8 -*-
"""
Created on Oct 4, 2020

@author: ahinoamp@gmail.com

Transfer specific history files to separate folder for processing

"""
import os
import re
import random   
import numpy as np
from shutil import copyfile

#Loop through all of the files and take the best and and another 5 random ones and copy them over
def GetThreadFromFile(file):
    result = re.search(r'\\'+'(.*)_G', file)
    return int(result.group(1))

def GetErrFromFile(file):
    result = re.search('Err_(.*).his', file)
    return float(result.group(1))

folderOut= 'HistoryPosterior/'
folderIn = 'HistoryFileTransferBest/'
from glob import glob
files = glob(folderIn+'*.his')
nFiles = len(files)

listHistoryNumbers= [2321, 2011, 3947, 4375, 2237, 1553, 1777, 1331, 503, 3139, 2857, 2587, 727, 4891, 1000, 1809, 728, 5133, 1116, 1651, 1891, 3711, 477, 2445, 3048, 3599, 3292, 3792, 4135, 1772, 452, 1944, 1007, 4212, 1593, 1253, 2675, 433, 3840, 4189, 2143, 2151, 7659, 3845, 1685, 1475, 4475, 3165, 1056, 3115, 940, 3561, 985, 9859, 1451, 764, 1603, 1049, 2217, 4887, 777, 2076, 2627, 3985, 4131, 400, 996, 3072, 1992, 4423, 3679, 4328, 2697, 2279, 2013, 392, 3155, 1057, 4711, 3200, 2509, 2021, 4335, 2184, 7323, 3223, 1244, 1064, 1737, 2227, 3575, 2520, 6899, 2593, 991, 2396, 1425, 3287, 1083, 3367, 2667, 1567, 4123, 3192, 575, 1208, 3756, 3673, 195, 3327, 2545, 2612, 2811, 2695, 1183, 712, 199, 2500, 1227, 3537, 1999, 4420, 3685, 4220, 1764, 3256, 3295, 2779, 4188, 3136, 1040, 1880, 3329, 2516, 1171, 3712, 416, 2979, 2023, 64, 1656, 2860, 2760, 1507, 9525, 804, 3937, 4883, 595, 3127, 3755, 2019, 3460, 3120, 4376, 4265, 671, 165, 2550, 3383, 405, 903, 2415, 2783, 1792, 2428, 3851, 2789, 4127, 4055, 3875, 2475, 1135, 688, 1708, 568, 176, 9861, 1363, 3936, 4112, 509, 3848, 3116, 3724, 1696, 4219, 756, 2248, 2552, 2920, 1317, 592, 557, 977, 716, 2513, 2020, 2618, 2480] 

MasterFileList = []
for j in range(len(listHistoryNumbers)):
    threadNum = listHistoryNumbers[j]
    files = glob(folderIn+'/'+str(threadNum)+'_G*.his')
    nFiles = len(files)   

    shortenName = re.search(r'\\'+'(.*)', file2keep).group(1)
    copyfile(file2keep, folderOut+shortenName) 
