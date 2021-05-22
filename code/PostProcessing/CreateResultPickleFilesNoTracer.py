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
import GeneralInversionUtil as GI
import LoadInputDataUtility as DI
import SimulationUtilities as sim
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
from deap import base

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
    x = Fault3DPoints[0]*P['cubesize']+P['xy_origin'][0]+P['xmin']+P['cubesize']
    y = Fault3DPoints[1]*P['cubesize']+P['xy_origin'][1]+P['ymin']+P['cubesize']
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
    

folder = 'HistoryFileTransferRandom/'

historyfiles = glob(folder+'*.his')
nFiles = len(historyfiles)
P={}


P['CalcGravity']=True
P['CalcMagnetics']=True
P['CalcGraniteTop']=True
P['CalcTracer']=False
P['CalcFaultIntersection']=True
P['DataTypes'] = ['Grav', 'Mag', 'GT', 'FaultMarkers']
P['JoinType']='LINES'
P['errCalcMethodFaultIntersection']='Distance'
P['MaxFaultIntersectionError'] = 500
P['HypP']={}
P['HypP']['Windows']=False
P['HypP']['graniteIdx'] = 4
P['HypP']['MaxFaultMarkerError'] = 525

#############################
## 0 define the model grid boundaries + some other input stuff
#############################  
 

folderoutput = folder+'Scratch/'
P['folder']=folderoutput

P['output_name'] = folderoutput+'noddy_out'
P['iterationNum'] =0


P['verbose']=True    
P['DatNormCoef'] = {'Grav': 2.4, 'Tracer': 1.0, 
                            'FaultMarkers': 500, 'GT': 315, 'Mag':300}
P['SimulationShiftType']= 'Median Datum Shift'
P['ErrorNorm']='L1'
P['ErrorType']='Global'
P['cubesize']=150
P['xy_origin']=[316448, 4379166, 1200-4000]
P['xy_extent'] = [8800, 9035,4000]
P['xmin'] = P['xy_origin'][0]
P['xmax'] = P['xy_origin'][0]+P['xy_extent'][0]

P['ymin'] = P['xy_origin'][1]
P['ymax'] = P['xy_origin'][1]+P['xy_extent'][1]

P['zmin'] = P['xy_origin'][2]
P['zmax'] = P['xy_origin'][2]+P['xy_extent'][2]

toolbox = base.Toolbox()
GI.register_sim_functions(P, toolbox)

P['toolbox']=toolbox

DI.loadData(P)

plt.close('all')
start = 0
ErrList = []
for h in range(nFiles):
    if(h<100):
        continue
    if(h>600):
        break
    print(h)
    hisfile = historyfiles[h]
    print(historyfiles[h])

    P['SampledInputFileName'] = hisfile
    #calculate model

    try:
        sim.simulate_calc_mismatch(P)
    
        print(list(P['FaultMarkers'].keys()))
    
    #    Err = GetErrFromFile(hisfile)
        Err = [P['Grav']['L1MismatchList'][-1],
               P['Mag']['L1MismatchList'][-1],
               P['GT']['L1MismatchList'][-1],
               P['FaultMarkers']['L1MismatchList'][-1]]
       
        
        ErrList.append(Err)
        FaultPriorMatrix = P['CombinedFaultMatrix'].astype(int)
        dictBlock = {}
        dictBlock['FaultBlock'] = FaultPriorMatrix
        dictBlock['FaultBlockErr'] = Err
    
            
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