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
from deap import base

import random
import pickle

def create_prior_models(values):
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
    HypPara['ScenarioNum'] = random.choice([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    HypPara['GeneralPerturbStrategy']=''
    HypPara['GlobalMoveEachDir'] = random.uniform(0, 800)
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
    
    try:
        P={}
        toolbox = base.Toolbox()
        GI.register_sim_functions(HypPara, toolbox)
        P['toolbox']=toolbox

        P['HypP'] = HypPara
        GI.setupFoldersParameters(P)    
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
        Err = [P['Grav']['L1MismatchList'][-1],
               P['Mag']['L1MismatchList'][-1],
               P['Tracer']['L1MismatchList'][-1],
               P['GT']['L1MismatchList'][-1],
               P['FaultMarkers']['L1MismatchList'][-1]]
       
        gravBlock = {}
        gravBlock['GravSim'] = P['Grav']['Sim_Obs']
        gravBlock['Err'] = Err
    
   
        with open('PriorFalsificationResults/gravfile_'+str(run_n)+'.pickle', 'wb') as handle:
            pickle.dump(gravBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        magBlock = {}
        magBlock['MagSim'] = P['Mag']['Sim_Obs']
        magBlock['Err'] = Err
    
        with open('PriorFalsificationResults/magfile_'+str(run_n)+'.pickle', 'wb') as handle:
            pickle.dump(magBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
        gtBlock = {}
        gtBlock['GTSim'] = P['GT']['Sim_Obs']
        gtBlock['Err'] = Err
    
        with open('PriorFalsificationResults/gtfile_'+str(run_n)+'.pickle', 'wb') as handle:
            pickle.dump(gtBlock, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    except:
        print('theres a problem in the air tonight')
    
if __name__== "__main__":
    params = {'run_n': 4458}
        
    create_prior_models(params)