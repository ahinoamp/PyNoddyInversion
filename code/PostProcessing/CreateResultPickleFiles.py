# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:04:00 2020

@author: ahinoamp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:26:56 2020

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

from sklearn.decomposition import PCA

def finalXYclean(x,y): 

    xy = np.concatenate((x.reshape(-1,1), y.reshape(-1,1)), axis=1)     
    # define a matrix
    pca = PCA(2)
    # fit on data
    pca.fit(xy)
       
    xypca = pca.transform(xy)

    newx = xypca[:,0]
    newy = xypca[:,1]

    indexSort = np.argsort(x)
    newx = newx[indexSort]
    newy = newy[indexSort]
    f = interpolate.interp1d(newx, newy, kind='linear')        
    newX=np.linspace(np.min(newx), np.max(newx), 100)
    newY = f(newX)            
    window = 43
    newY = savgol_filter(newY, window, 2)

    xyclean = pca.inverse_transform(np.concatenate((newX.reshape(-1,1), newY.reshape(-1,1)), axis=1) )

    xc=xyclean[:,0]
    yc = xyclean[:,1]
    return xc, yc

def getZSection(FaultsXY, i, P, sliceValue):
    fn = P['faultNumbers'][i]
  
    minSlice = sliceValue-150
    maxSlice = sliceValue+150
    
    filterFault = FaultsXY[str(fn)]==True
    x = FaultsXY.loc[filterFault, 'x'].values
    y = FaultsXY.loc[filterFault, 'y'].values
    z = FaultsXY.loc[filterFault, 'z'].values
    
    filterV = (z<maxSlice)&(z>minSlice)

    x = x[filterV]
    y = y[filterV]
    z = z[filterV]

    if(len(x)<2):
        return [0]
    
    x,y = finalXYclean(x,y)
    return x,y
    
def getXSection(FaultsXY, i, P, sliceValue):
    fn = P['faultNumbers'][i]
  
    minSlice = sliceValue-150
    maxSlice = sliceValue+150
    
    filterFault = FaultsXY[str(fn)]==True
    x = FaultsXY.loc[filterFault, 'x'].values
    y = FaultsXY.loc[filterFault, 'y'].values
    z = FaultsXY.loc[filterFault, 'z'].values
    
    filterV = (x<P['xy_origin'][0]+maxSlice)&(x>P['xy_origin'][0]+minSlice)

    x = x[filterV]
    y = y[filterV]
    z = z[filterV]

    if(len(x)<2):
        return [0]
    y,z = finalXYclean(y,z)

    return y,z


def getYSection(FaultsXY, i, P, sliceValue):
    fn = P['faultNumbers'][i]
  
    minSlice = sliceValue-150
    maxSlice = sliceValue+150
    
    filterFault = FaultsXY[str(fn)]==True
    x = FaultsXY.loc[filterFault, 'x'].values
    y = FaultsXY.loc[filterFault, 'y'].values
    z = FaultsXY.loc[filterFault, 'z'].values
    
    filterV = (y<P['xy_origin'][1]+maxSlice)&(y>P['xy_origin'][1]+minSlice)

    x = x[filterV]
    y = y[filterV]
    z = z[filterV]

    if(len(x)<2):
        return [0]

    x,z = finalXYclean(x,z)

    return x,z
                 
def GetFaultXYCode(P):
    Fault3DPoints = np.nonzero(P['CombinedFaultMatrix']>-0.1)
    x = Fault3DPoints[0]*P['cubesize']+P['xy_origin'][0]+P['N1'].xmin+P['cubesize']
    y = Fault3DPoints[1]*P['cubesize']+P['xy_origin'][1]+P['N1'].ymin+P['cubesize']
#    z = (P['xy_origin'][2]+P['N1'].zmax) - Fault3DPoints[2]*P['cubesize']-P['cubesize']
    z = -2700 + Fault3DPoints[2]*float(P['cubesize'])+P['cubesize']
    faultcodes = P['CombinedFaultMatrix'][Fault3DPoints]
    
    FaultXYCode = pd.DataFrame({'x': x, 'y':y, 'z': z, 'codes': faultcodes})
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    z= z.reshape(-1,1)
    code = faultcodes.reshape(-1,1)

    faultNumbers = list(np.unique(sum(P['zEdgeDict'].values(), [])))
    faultNumbers.remove(-1)
    
    codePerFault = flipdict(P['zEdgeDict'])
    nFaults = len(faultNumbers)
    for i in range(nFaults):
        codesFaultI = codePerFault[faultNumbers[i]]        
        indexFault = np.isin(faultcodes, codesFaultI)
        FaultXYCode[str(i)]=indexFault

    P['faultNumbers'] = np.arange(nFaults)
    return FaultXYCode

def flipdict(dict2flip):
    
    flippedDict = {}
    for key, value in dict2flip.items():
        for fn in value:
            if(fn in list(flippedDict.keys())):
                flippedDict[fn].append(key)
            else:
                flippedDict[fn]=[key]
    return flippedDict
    

folder = 'HistoryFileTransfer/'

historyfiles = glob(folder+'*.his')
nFiles = len(historyfiles)
P={}


P['CalcGravity']=True
P['CalcMagnetics']=True
P['CalcGraniteTop']=True
P['CalcTracer']=True
P['CalcFaultIntersection']=True
P['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultIntersection']
P['JoinType']='LINES'
P['errCalcMethodFaultIntersection']='Distance'
P['MaxFaultIntersectionError'] = 500

#############################
## 0 define the model grid boundaries + some other input stuff
#############################  
 

folderoutput = folder+'Scratch4/'
P['folder']=folderoutput

P['output_name'] = folderoutput+'noddy_out'
P['iterationNum'] =0

Norm = pd.read_pickle('NormalizingFactorSummary.pkl')

P['verbose']=True    
P['Grav_comboNormalizingFactor'] = Norm['Grav']
P['Mag_comboNormalizingFactor'] = Norm['Mag']
P['Tracer_comboNormalizingFactor'] = Norm['Tracer']
P['GT_comboNormalizingFactor'] = Norm['GT']
P['FaultIntersection_comboNormalizingFactor'] = Norm['FaultIntersection']    
P['SimulationShiftType']='Datum Shift'
P['ErrorNorm']='L1'
P['ErrorType']='Global'
P['cubesize']=150
P['xy_origin']=[316448, 4379166, 1200-4000]
P['xy_extent'] = [8800, 9035,4000]

DI.loadData(P)

plt.close('all')
start = 0
ErrList = []
for h in range(nFiles):
    print(h)
    hisfile = historyfiles[h]
    print(historyfiles[h])

    P['SampledInputFileName'] = hisfile
    #calculate model

    try:
        GI.SimulateGetGlobalMismatch(P)
    
    #    Err = GetErrFromFile(hisfile)
        Err = [P['Grav_MismatchList'][-1],
               P['Mag_MismatchList'][-1],
               P['Tracer_MismatchList'][-1],
               P['GT_MismatchList'][-1],
               P['FaultIntersection_MismatchList'][-1]]
       
        ErrList.append(Err)
        FaultPriorMatrix = P['CombinedFaultMatrix'].astype(int)
        dictBlock = {}
        dictBlock['FaultBlock'] = FaultPriorMatrix
        dictBlock['FaultBlockErr'] = Err
    
        with open(folder+'Blocks/file_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(dictBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        Lithology = {}
        Lithology['Lithology'] = P['N1'].block
        Lithology['LithologyErr'] = Err
    
        with open(folder+'Lithology/file_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(Lithology, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #get fault location 
        dictFaultI = {}
        FaultsXY = GetFaultXYCode(P)
        dictFaultI['nFaults'] = len(P['faultNumbers'])
        dictFaultI['Err']=Err
        for i in range(len(P['faultNumbers'])):
            a = getXSection(FaultsXY, i, P, P['xy_extent'][0]/2.0) 
            b = getYSection(FaultsXY, i, P, P['xy_extent'][1]/2.0) 
            c = getZSection(FaultsXY, i, P, 0)  
            if((len(a)>=2)|(len(b)>=2)|(len(c)>=2)):
                smallDict = {}
                if(len(a)>=2):
                    smallDict['X']=a
                if(len(b)>=2):
                    smallDict['Y']=b
                if(len(c)>=2):
                    smallDict['Z']=c
                        
                dictFaultI[str(i)]=smallDict
    
        with open(folder+'Faults/file_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(dictFaultI, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        continue