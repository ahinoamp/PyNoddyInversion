# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:58:21 2020

@author: ahinoamp
"""

import GeneralInversionUtil as GI
import VisualizationUtilities as Viz
import PriorUncertaintyUtil as Unc
import LoadInputDataUtility as DI
import SimulationUtilities as sim
import SamplingHisFileUtil as sample
import PSO_GA_Utilities as PSOGA
import MCMC_Util as MCMC

import pandas as pd
import SimulationUtilities as sim
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

def create_prior_models(values):
    startN = 
    endN = 
    
    P={}
    P['CalcGravity']=True
    P['CalcMagnetics']=True
    P['CalcGraniteTop']=True
    P['CalcTracer']=True
    P['CalcFaultIntersection']=True
    P['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultMarkers']
    P['JoinType']='LINES'
    P['errCalcMethodFaultIntersection']='Distance'
    P['MaxFaultIntersectionError'] = 500
    
    #############################
    ## 0 define the model grid boundaries + some other input stuff
    #############################  
    folder = 'Results/'
     
    folderoutput = folder+'Scratch/'
    P['folder']=folderoutput
    P['output_name'] = P['folder']+'noddy_out'
    P['iterationNum'] =0
    
    
    Norm = {'Grav': 2.4, 'Tracer': 1.0, 'FaultMarkers': 500, 'GT': 315, 'Mag':300}
    
    P['verbose']=True     
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
    for i in range(startN, endN):
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
            P['ScenarioNum'] = random.choice([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
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
    
            DI.loadData(P)
            
            #############################
            ## 1.5 Define the range of uncertainties for these runs
            ## 1.75 choose only a subset of events which to care about
            ################################
            P['ModelParamTable'] = GI.InitializeParameters(P)
            
        
            #############################
            ## 2. Calculate mismatch #1
            #############################
            P['iterationNum']=0
            MCMC.updateDataTypeWeights(P)
            sample.ProposeParameters(P)
            sim.simulate_calc_mismatch(P)
    
            #    Err = GetErrFromFile(hisfile)
            Err = [P['Grav_MismatchList'][-1],
                   P['Mag_MismatchList'][-1],
                   P['Tracer_MismatchList'][-1],
                   P['GT_MismatchList'][-1],
                   P['FaultIntersection_MismatchList'][-1]]
           
            ErrList.append(Err)
            gravBlock = {}
            gravBlock['Grav']['Sim'] = P['gSim']
            gravBlock['Err'] = Err
        
            with open('Z:/OptimisationPatua/ReducedDimensionSpace2/gravfile_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(gravBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            magBlock = {}
            magBlock['Mag']['Sim'] = P['gSimMag']
            magBlock['Err'] = Err
        
            with open('Z:/OptimisationPatua/ReducedDimensionSpace2/magfile_'+str(i)+'.pickle', 'wb') as handle:
                pickle.dump(magBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
            gtBlock = {}
            gtBlock['GT']['Sim'] = P['GT_Sim']
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
    
