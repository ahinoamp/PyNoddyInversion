# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:58:21 2020

@author: ahinoamp
"""

import GeneralInversionUtil as GI
import LoadInputDataUtility as DI
import SimulationUtilities as sim
import SamplingHisFileUtil as sample
import MCMC_Util as MCMC
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.signal import savgol_filter
import from glob import glob

from deap import base

import random
import pickle

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
    
    filterV = (x<P['HypP']['xy_origin'][0]+maxSlice)&(x>P['HypP']['xy_origin'][0]+minSlice)

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
    
    filterV = (y<P['HypP']['xy_origin'][1]+maxSlice)&(y>P['HypP']['xy_origin'][1]+minSlice)

    x = x[filterV]
    y = y[filterV]
    z = z[filterV]

    if(len(x)<2):
        return [0]

    x,z = finalXYclean(x,z)

    return x,z
                 
def GetFaultXYCode(P):
    Fault3DPoints = np.nonzero(P['CombinedFaultMatrix']>-0.1)
    x = Fault3DPoints[0]*P['cubesize']+P['xmin']+P['cubesize']/2.
    y = Fault3DPoints[1]*P['cubesize']+P['ymin']+P['cubesize']/2.
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

def create_lineaments2(values):
    folder = 'Scratch2/'

    run_n = values['run_n']
    
    HypPara={}
    HypPara['thread_num']=run_n
    HypPara['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultMarkers']
    HypPara['JoinType']='LINES'
    HypPara['errCalcMethodFaultIntersection']='Distance'
    HypPara['MaxFaultMarkerError'] = 525
    HypPara['BaseFolder']='PriorFalsificationResults/'
    HypPara['output_name'] = HypPara['BaseFolder']+'noddy_out'
    HypPara['iterationNum'] =0
        
    HypPara['verbose']=False
    HypPara['GeneralPerturbStrategy']='OnlyGlobal'
    HypPara['ControlPointMovementUnits'] = 'Absolute value'
    HypPara['nruns']=1
    HypPara['xy_origin']=[316448, 4379166, -2700]
    HypPara['xy_extent'] = [8850, 9000,3900]
    HypPara['DatNormCoef'] = {'Grav': 2.4, 'Tracer': 1.0, 
                            'FaultMarkers': 500, 'GT': 315, 'Mag':300}
    HypPara['verbose']=True    
    HypPara['graniteIdx'] = 4
    HypPara['Windows'] = True
    
    HypPara['verbose']=True     
    HypPara['SimulationShiftType']='Median Datum Shift'
    HypPara['ErrorNorm']='L1'
    HypPara['ErrorType']='Global'
    HypPara['DatNormMethod']='Given'
    HypPara['cubesize']=150

    
    HypPara['Toy']=False
    HypPara['cubesize'] = 150
    HypPara['ScenarioNum'] = random.choice([11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    HypPara['GeneralPerturbStrategy']=''
    HypPara['GlobalMoveEachDir'] = random.uniform(300, 900)
    HypPara['XYZ_Axes_StepStd'] = random.uniform(70, 150)
    HypPara['Dip_StepStd']= random.uniform(2, 7)
    HypPara['Slip_StepStd']=random.uniform(30, 90)
    HypPara['localWeightRadiusMult']=random.uniform(1, 2.5)
    HypPara['DipDirection_StepStd']=random.uniform(2, 10)
    HypPara['SlipParam']=random.uniform(0.09, 0.22)
    HypPara['SteppingSizeMultiplierGlobal']= random.uniform(1.0/1.1, 1.0/0.7)
    HypPara['SteppingSizeMultiplierLocal']=random.uniform(1.0/1.1, 1.0/0.7)
    HypPara['AmplitudeRatioChange']=random.uniform(0.05, 0.25)
    HypPara['AzimuthMoveEachDirection']=random.uniform(3, 10)
    HypPara['AxisRatioChange']=random.uniform(0.02, 0.25)
    HypPara['DipMoveEachDirection']=random.uniform(25, 40)
    HypPara['StartUpdateLocalWeight']=random.uniform(30, 60)
    HypPara['UpdateLocalWeightsFreqRuleBased']=random.uniform(1, 40)
    HypPara['verbose']=True    
    HypPara['MO_WeightingMethod']='Equal'    
    ##############################
    ## Setup the folders
    ##############################
    
#    try:
    P={}
    toolbox = base.Toolbox()
    GI.register_sim_functions(HypPara, toolbox)
    P['toolbox']=toolbox

    P['HypP'] = HypPara
    GI.setupFoldersParameters(P)    
    DI.loadData(P)
       
    #############################
    ## 2. Calculate mismatch #1
    #############################
    P['iterationNum']=0
    MCMC.updateDataTypeWeights(P)

    P['SampledInputFileName'] = values['hisfile']

    sim.simulate_calc_mismatch(P)

    #    Err = GetErrFromFile(hisfile)
    Err = [P['Grav']['L1MismatchList'][-1],
           P['Mag']['L1MismatchList'][-1],
           P['Tracer']['L1MismatchList'][-1],
           P['GT']['L1MismatchList'][-1],
           P['FaultMarkers']['L1MismatchList'][-1]]
   
    FaultPriorMatrix = P['CombinedFaultMatrix'].astype(int)
    dictBlock = {}
    dictBlock['FaultBlock'] = FaultPriorMatrix
    dictBlock['FaultBlockErr'] = Err

    with open(folder+'Blocks/file_'+str(run_n)+'.pickle', 'wb') as handle:
        pickle.dump(dictBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
       
    #get fault location 
    dictFaultI = {}
    FaultsXY = GetFaultXYCode(P)
    dictFaultI['nFaults'] = len(P['faultNumbers'])
    dictFaultI['Err']=Err
    for i in range(len(P['faultNumbers'])):
        a = getXSection(FaultsXY, i, P, P['HypP']['xy_extent'][0]/2.0)
        b = getYSection(FaultsXY, i, P, P['HypP']['xy_extent'][1]/2.0)
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
    
        with open(folder+'Faults/file_'+str(run_n)+'.pickle', 'wb') as handle:
            pickle.dump(dictFaultI, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#    except:
#        print('theres a problem in the air tonight')
    
if __name__== "__main__":
    params = {'run_n': 4458, 'hisfile': 'faultmodel.his'}

    create_lineaments2(params)