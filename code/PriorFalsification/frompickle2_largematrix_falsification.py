# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:52:30 2020

@author: ahinoamp
"""
import numpy as np
from glob import glob
import pandas as pd
import pickle

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

folder = 'PriorFalsificationResults/'
DataTypes = ['grav', 'mag', 'gt']
dictName = ['GravSim', 'MagSim', 'GTSim']
datasize = [337, 743, 32]
for d in range(3):
    DataType=DataTypes[d]
    picklefiles = glob(folder+DataType+'*.pickle')
    nFiles = len(picklefiles)
    
    started=0
    numPrior = nFiles
    ErrList=[]
    for i in range(nFiles):    
    #for i in range(500):    
    
        print('reading: '+picklefiles[i])
    
        with open(picklefiles[i], 'rb') as handle:
            dictFaultI = pickle.load(handle)
            
        info = dictFaultI[dictName[d]]
        if(np.shape(info)[0]!=datasize[d]):
            print('problem : ' +str(np.shape(info)))
            continue
        
        if(i==0):
            largeMatrix = info.reshape(-1,1)
        else:
            largeMatrix = np.concatenate((largeMatrix, info.reshape(-1,1)), axis=1)
            
    np.savetxt(DataType+'_largeMatrix.csv', largeMatrix, delimiter=',', fmt='%.4f') 

            
