# -*- coding: utf-8 -*-
"""
Created on Oct. 4, 2020

@author: ahinoamp@gmail.com

This script copies a portion of the paramter history into a separate file for 
further analysis

"""
import os
import re
import random   
import numpy as np
from shutil import copyfile
import pandas as pd


Norm = pd.read_pickle('NormalizingFactorSummary.pkl')
folder= 'Combo_Scratch/'

folders = [os.path.isdir(folder+'/'+i) for i in os.listdir(folder)]

from glob import glob
folders = glob("Combo_Scratch/*/")
nFolders = len(folders)

MasterDataList = []
firstRun=0
for i in range(nFolders):
    threadfolder = folders[i]
    print('reading: '+ threadfolder)
    parameterhistoryFiles = glob(threadfolder+"ParameterHistory.csv")
    if(len(parameterhistoryFiles)>0):
        parameterhistoryf = parameterhistoryFiles[0]
    else:
        continue
    result = re.search('Thread(.*)/', parameterhistoryf)
    threadnum = int(result.group(1))


    #read in the file
    PH = pd.read_csv(parameterhistoryf)
    
    oldColumnNames = list(PH.keys())
    newColumnNames = []
    for c in range(len(oldColumnNames)):
        oldname = oldColumnNames[c]
        try:
            if (('Fault' in oldname) and ('Error' not in oldname)):
                parts = oldname.split('_')
                newname = parts[0]+'_'+parts[1]+'_'+parts[3]
                newColumnNames.append(newname)
            else:
                newColumnNames.append(oldname)
        except:
            print(oldname)
            
    PH.columns = newColumnNames
    PH['filename']=threadfolder
    PH['threadnum']=threadnum
    #Get the parameters for the row with the lowest error
#    LowestError = PH[PH['NewError'] == np.min(['NewError'])]

    targetNRow = 4.
    nSamples = int(np.max([(len(PH)/targetNRow),1]))
    Subsampled = PH.iloc[::nSamples, :]   
    MasterDataList.append(Subsampled)

BigPD = pd.concat(MasterDataList, sort=False)
BigPD.to_csv('ErrorHistorySummary.csv')

