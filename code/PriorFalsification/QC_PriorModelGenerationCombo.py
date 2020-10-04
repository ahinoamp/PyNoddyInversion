# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:58:21 2020

@author: ahinoamp
"""

import pandas as pd
import GravityInversionUtilities as GI
import VisualizationUtilities as Viz
import PriorUncertaintyUtil as Unc
import LoadInputDataUtility as DI
import pynoddy, time
import matplotlib.pyplot as plt
import pynoddy.history
import pynoddy.experiment
import pynoddy.events
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import random
import pickle

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
folder = 'Z:/OptimisationPatua/CheckHistory/'
 
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
P['xy_origin']=[316448, 4379166, 1200-4000]
P['xy_extent'] = [8800, 9035,4000]

DI.loadData(P)

##############################
## Setup the folders
##############################

nRealizations = 5500
ErrList=[]
for i in range(nRealizations):
    if(i<288):
        continue
    print(i)
    try:
        SampledInputFileName = folder+'HistoryTest'+str(i)+'.his'
        xy_origin=[316448, 4379166, 1200-4000]
        xy_extent = [8800, 9035,4000]
        
        ModelParametersTableF ='scratchmodelparameterstable.csv'
        P['xy_extent']=xy_extent
        P['SampledInputFileName']=SampledInputFileName
        P['Toy']=False
        P['GlobalMoveEachDir'] = 1000
        P['XYZ_Axes_StepStd'] = 300
        P['cubesize'] = 150
        P['ScenarioNum'] = random.choice([5, 10,10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19])
        P['SlipParam'] = 0.1
        P['xy_origin']=xy_origin
        P['GeneralPerturbStrategy']=''
        P['GlobalMoveEachDir'] = random.uniform(0, 800)
        P['XYZ_Axes_StepStd'] = random.uniform(70, 150)
        P['Dip_StepStd']= random.uniform(2, 7)
        P['Slip_StepStd']=random.uniform(30, 90)
        P['localWeightRadiusMult']=random.uniform(1, 2.5)
        P['DipDirection_StepStd']=random.uniform(5, 20)
        P['SlipParam']=random.uniform(0.07, 0.1)
        P['SteppingSizeMultiplierGlobal']= random.uniform(1.0/1.1, 1.0/0.7)
        P['SteppingSizeMultiplierLocal']=random.uniform(1.0/1.1, 1.0/0.7)
        P['AmplitudeRatioChange']=random.uniform(0.05, 0.25)
        P['AzimuthMoveEachDirection']=random.uniform(3, 15)
        P['AxisRatioChange']=random.uniform(0.02, 0.25)
        P['DipMoveEachDirection']=random.uniform(15, 28)
        P['StartUpdateLocalWeight']=random.uniform(30, 60)
        P['UpdateLocalWeightsFreqRuleBased']=random.uniform(1, 40)
        P['verbose']=True
        
        Unc.DefinePriorUncertainty(P, ModelParametersTableF)
        
        ################################
        ## 1.75 choose only a subset of events which to care about
        ################################
        ModelParametersTable = pd.read_csv(ModelParametersTableF)
        ModelParametersTable = GI.SelectFaultingEvents(ModelParametersTable, P)        
        ModelParametersTable = GI.OrderFaultingEvents(ModelParametersTable,P)        
        
        ModelParametersTable.to_csv(folder+'ThreadModelParameters.csv')
        FullParameterHistory = ModelParametersTable.copy()
        P['FullParameterHistory']=FullParameterHistory
        P['nEvents']=int(np.max(ModelParametersTable['EventNumber']))
        P['nFaults']=int(np.sum(ModelParametersTable['EventName'].drop_duplicates().str.contains('Fault')))
        ##############################
        ##1.8 Create a base file and remove from table all parameters that have no std
        ##############################
        #debugging line
        #ModelParametersTable.loc[15:, 'proposal_std_or_range']=0
        GI.CreateTemplatePyNoddyFile(P,ModelParametersTable)
        
        #only keep those parameters that need to be optimized in the model parameter table
        P['OptimizeParametersIdx'] = ModelParametersTable['proposal_std_or_range']>0.000001
        ModelParametersTable=ModelParametersTable[ModelParametersTable['proposal_std_or_range']>0.000001].reset_index()
        ModelParametersTable = ModelParametersTable.drop(['index'], axis=1)
        P['nParameters']= np.shape(ModelParametersTable)[0]
        P['ModelParametersTable']=ModelParametersTable
        
        
        #############################
        ## 2. Calculate mismatch #1
        #############################
        P['iterationNum']=0
        P['GeneralPerturbStrategy']=''
        P['ErrorType']='Global'
        P['LearningRate']='None'
        P['sampledParamterValues']=GI.ProposeParametersExploration(P)
        P['ModelParametersTable'][str(P['iterationNum'])] = P['sampledParamterValues']
        P['Grav_AllAcceptedList'] = [1]
        P['GT_AllAcceptedList'] = [1]
        P['Mag_AllAcceptedList'] = [1]
        P['Tracer_AllAcceptedList'] = [1]
        P['FaultIntersection_AllAcceptedList'] = [1]
        P['Combo_AllAcceptedList']=[1]
        P['Combo_MismatchList'] = [2.2]
        P['OptimMethod']='MCMC'
        P['SimulationsExplorationStage']=2
        P['ExplorationStage']='Explore'
        P['Combo_lastAcceptedIndex']=0
        P['AcceptProbType']=''    
        P['ErrorType']='Global'
        P['Combo_lastAcceptanceProbability']=0.3
        P['Combo_lastNormalizingFactor']=0.1
        P['thread_num']=i
        
        GI.SimulateGetGlobalMismatch(P)
        GI.OutputImageAndHistoryFile(P, folderoutput+'Visualization/Viz_'+str(i))
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
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/gravfile_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(gravBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        magBlock = {}
        magBlock['magSim'] = P['gSimMag']
        magBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/magfile_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(magBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        gtBlock = {}
        gtBlock['gtSim'] = P['GT_Sim']
        gtBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/gtfile_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(gtBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        tracerBlock = {}
        tracerBlock['tracerSim'] = P['TracersConnected']
        tracerBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/tracerfile_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(tracerBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        faultBlock = {}
        faultBlock['faultSim'] = P['simDistance2Intersection']
        faultBlock['Err'] = Err
    
        with open('Z:/OptimisationPatua/ReducedDimensionSpace2/faultfile_'+str(i)+'.pickle', 'wb') as handle:
            pickle.dump(gtBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except:
        print('theres a problem in the air tonight')
        continue

