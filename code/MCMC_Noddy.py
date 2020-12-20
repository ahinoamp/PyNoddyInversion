# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 12:03:16 2020

@author: ahinoam p
/"""
#############################
## Import block
#############################
import pandas as pd
import numpy as np
import time
from deap import base

import GeneralInversionUtil as GI
import VisualizationUtilities as Viz
import LoadInputDataUtility as DI
import SimulationUtilities as sim
import SamplingHisFileUtil as sample
import MCMC_Util as MCMC

def MCMC_Noddy(HypPara):

    toolbox = base.Toolbox()
    GI.register_sim_functions(HypPara, toolbox)
    register_functions_MCMC(HypPara, toolbox)   
    
    P={}
    P['toolbox']=toolbox
    P['HypP'] = HypPara
    #1. Setup output folders and initialize parameters
    GI.setupFoldersParameters(P)    
    
    #############################
    ## 1. load real observed data
    #############################
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

    if(P['HypP']['ErrorType']=='Local'):
        GI.UpdateLocalWeights(P)
    
    #Book keeping  
    MCMC.AcceptFirstRound(P)

    GI.OutputImageAndHisFile(P)

    #############################
    ## 1.9. An exploration stage
    #############################
    if(P['HypP']['ExplorationStage']=='Explore'):
        MCMC.explore_space(P)
        
    #############################
    ## 3. MCMC loop
    #############################
    
    start = time.time()
    for i in range(1, P['nRuns']):
        P['iterationNum']=P['iterationNum']+1
        if(P['verbose']):
            print('Run Number: ' +str(P['iterationNum']))

        #############################
        ## 3.2 Run the next simulation
        #############################            
        sample.ProposeParameters(P)
        sim.simulate_calc_mismatch(P)
                
        #############################
        ## 3.2 Update local Weights if necessary
        #############################
        if((P['HypP']['ErrorType']=='Local') & (P['HypP']['LocalWeightsMode']=='Many')):
            GI.UpdateLocalWeights(P)

        #############################
        ## 3.1 Check acceptance global and local methods
        #############################
        toolbox.check_acceptance(P)

        if(np.mod((P['iterationNum']),P['HypP']['OutputImageFreq'])==0):
            GI.OutputImageAndHisFile(P)

        #############################
        ## 3.3 Check whether to stop early
        #############################
        breakEarly = GI.CheckEarlyStop(P)
        if(breakEarly==1):
            break
           
    #############################
    ## 3.3 Save data/results/parameters to files
    #############################
    GI.SaveResults2Files(P)
    
    #############################
    ## 3.3 visuals
    #############################
    Viz.GenerateVisualizations(P)


    end = time.time()
    if(P['verbose']):
        print('Calculation time took '+str(end - start) + ' seconds')
        print('Calculation took '+str((end - start)/P['nRuns']) + ' seconds per realization')
    
    return 1

def register_functions_MCMC(P, toolbox):
    '''Create a toolbox of functions
       This way, the function always has the same name, but can have 
       different content. Allows to keep the same algorithm structure, but also
       test some interesting variations.'''

    if(P['ErrorType']=='Global'):
        toolbox.register("check_acceptance", MCMC.check_acceptance_global)
    else: #local
        toolbox.register("check_acceptance", MCMC.check_acceptance_local)
        

if __name__== "__main__":
    start = time.time()
    tasks = pd.read_pickle('Combo_Scratch/parameters.pkl')
    params = tasks.iloc[91, :]
    params = params.to_dict()
    params['OutputImageFreq']=15
    params['verbose']=True    
    params['Windows'] = True
        
    MCMC_Noddy(params)
    
    end = time.time()
    if(params['verbose']):
        print('Calculation time took '+str(end - start) + ' seconds')