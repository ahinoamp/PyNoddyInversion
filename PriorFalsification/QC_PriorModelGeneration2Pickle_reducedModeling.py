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

folder = 'Z:/OptimisationPatua/CheckHistory/'

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
 
folderoutput = folder+'Scratch/'
P['folder']=folderoutput
P['output_name'] = P['folder']+'noddy_out'
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
P['xy_origin']=[316448, 4379166, -2700]
P['xy_extent'] = [8800, 9035,4000]

DI.loadData(P)

plt.close('all')
start = 0
ErrList = []
for h in range(nFiles):
    if(h<0):
        continue
    print(h)
    hisfile = historyfiles[h]
    print(historyfiles[h])

    try:
        P['SampledInputFileName'] = hisfile
        #calculate model
        GI.SimulateGetGlobalMismatch(P)
        GI.OutputImageAndHistoryFile(P, folderoutput+'Visualization/Viz_'+str(h))
    #    Err = GetErrFromFile(hisfile)
        Err = [P['Grav_MismatchList'][-1],
               P['Mag_MismatchList'][-1],
               P['Tracer_MismatchList'][-1],
               P['GT_MismatchList'][-1],
               P['FaultIntersection_MismatchList'][-1]]
       
        ErrList.append(Err)
        gravBlock = {}
        gravBlock['gravSim'] = P['gSim']
        gravBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/gravfile_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(gravBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        magBlock = {}
        magBlock['magSim'] = P['gSimMag']
        magBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/magfile_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(magBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        gtBlock = {}
        gtBlock['gtSim'] = P['GT_Sim']
        gtBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/gtfile_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(gtBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        tracerBlock = {}
        tracerBlock['tracerSim'] = P['TracersConnected']
        tracerBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/tracerfile_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(tracerBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        faultBlock = {}
        faultBlock['faultSim'] = P['simDistance2Intersection']
        faultBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/faultfile_'+str(h)+'.pickle', 'wb') as handle:
            pickle.dump(gtBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('Theres a problem in the air tonight')
        continue