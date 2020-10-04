# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 13:19:50 2020

@author: ahinoamp@gmail.com

Save simulated data from single model run for prior falsification analysis
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


folder = 'Z:/TransferThesisDraft/HistoryFileTransfer/'

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
 
folderoutput = folder+'Scratch3/'
P['folder']=folderoutput
P['output_name'] = P['folder']+'noddy_out'
P['iterationNum'] =0


Norm = pd.read_pickle('NormalizingFactorSummary.pkl')

P['verbose']=False    
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
DataTypes = ['grav', 'mag', 'gt']

folder = 'Z:/OptimisationPatua/ReducedDimensionSpace/'
file = folder+'dataObs.csv'

data_mat = (P['gObs']).reshape(-1,1)
DataType = DataTypes[0]
np.savetxt(folder+DataType+'_ObslargeMatrix.csv', data_mat, delimiter=',', fmt='%.4f') 

datamag = (P['MagObs']).reshape(-1,1)
DataType = DataTypes[1]
np.savetxt(folder+DataType+'_ObslargeMatrix.csv', datamag, delimiter=',', fmt='%.4f') 

datagt=(P['GT_Obs']).reshape(-1,1)
DataType = DataTypes[2]
np.savetxt(folder+DataType+'_ObslargeMatrix.csv', datagt, delimiter=',', fmt='%.4f') 


