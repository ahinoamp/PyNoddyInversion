# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:52:30 2020

@author: ahinoamp
"""
import GravityInversionUtilities as GI
import LoadInputDataUtility as DI
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
from scipy import interpolate
import scipy as sp
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import re
from scipy.signal import savgol_filter
from matplotlib.lines import Line2D
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import random

P={}
xy_origin=[316448, 4379166, 1200-4000]
xy_extent = [8800, 9035,4000]
P['xy_origin']=xy_origin
P['xy_extent'] = xy_extent
cube_size = 150

xsection = P['xy_extent'][0]/2.0 
ysection = P['xy_extent'][1]/2.0 
zsection = 0 
        
fz = 14

folder = 'Z:/OptimisationPatua/ReducedDimensionSpace2/'

Norm = pd.read_pickle('NormalizingFactorSummary.pkl')
DataTypes = ['grav', 'mag', 'gt']
dictName = ['gravSim', 'magSim', 'gtSim']

for d in range(3):
    DataType=DataTypes[d]
    picklefiles = glob(folder+DataType+'*.pickle')
    nFiles = len(picklefiles)
    nFiles = 1590
    
    started=0
    numPrior = nFiles
    ErrList=[]
    for i in range(nFiles):    
    #for i in range(500):    
    
        print('reading: '+picklefiles[i])
    
        with open(picklefiles[i], 'rb') as handle:
            dictFaultI = pickle.load(handle)
            
        info = dictFaultI[dictName[d]]
        Err = dictFaultI['Err']
        NormErr = (Err[0]/Norm['Grav'] + 
                   Err[1]/Norm['Mag'] +
                   Err[2]/Norm['Tracer'] +
                   Err[3]/Norm['GT'] +
                   Err[4]/Norm['FaultIntersection'])/5.0
        
        ErrList.append(NormErr)    
    
        if(i==0):
            largeMatrix = info.reshape(-1,1)
        else:
            largeMatrix = np.concatenate((largeMatrix, info.reshape(-1,1)), axis=1)
            
    np.savetxt(folder+DataType+'_largeMatrix.csv', largeMatrix, delimiter=',', fmt='%.4f') 
    
errPD=pd.DataFrame({'ErrNorm': ErrList})
errPD.to_csv('ErrNorm.csv')
            
