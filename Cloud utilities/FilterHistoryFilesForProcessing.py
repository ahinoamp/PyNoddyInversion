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

folderOut= 'Z:/TransferThesisDraft/ReadyHistory/'
folderIn = 'Z:/TransferThesisDraft/HistoryFileTransfer/'
from glob import glob
files = glob(folderIn+'*.his')
nFiles = len(files)

listHistoryNumbers= [824,1154,112,2031,1260,810,848,98,2010,1964,1522,573,1422,136,1428,155,1650,671,1475,1116,1106,1931,1962,1540,578,1824,1564,327,1859,1270]

MasterFileList = []
for j in range(len(listHistoryNumbers)):
    threadNum = listHistoryNumbers[j]
    files = glob(folderIn+r'\\'+str(threadNum)+'_G*.his')
    nFiles = len(files)   
    ErrV=1000000.0
    for i in range(nFiles):
        file_i = files[i]
        print('reading: '+ file_i)
        Err=GetErrFromFile(file_i)
        if(Err<ErrV):
            ErrV = Err
            file2keep = file_i

    shortenName = re.search(r'\\'+'(.*)', file2keep).group(1)
    copyfile(file2keep, folderOut+shortenName) 
