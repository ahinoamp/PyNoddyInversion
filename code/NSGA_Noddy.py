# -*- coding: utf-8 -*-
"""
Created on September 16, 2020

@author: ahinoamp@gmail.com

This script provides the Non Dominated Sorting Genetic Algorithm (NSGA) for 
inverting sturctural geological models created by PyNoddy for matching typical 
geothermal data sets. 

This script is based on code from the deap project.
"""
import random
import numpy

import PSO_GA_Utilities as GA_Util
import LoadInputDataUtility as DI
import GeneralInversionUtil as GI
import SamplingHisFileUtil as sample
import SimulationUtilities as sim

import numpy as np
import pandas as pd

import deap_base as base
import deap_creator as creator
import deap_init as init
import deap_emo as emo


def NSGA2_Noddy(HypP):
    
    # set up a dictionary to hold all important information, including the 
    # hyper parameters
    P={}
    P['HypP'] = HypP
    
    #1. Setup output folders and initialize parameters
    GI.setupFoldersParameters(P)    

    #2. Load the different observed data types       
    DI.loadData(P)

    #3. Set up a table where the geological model parameters are stored 
    P['ModelParamTable'] = GI.InitializeParameters(P)

    #4. Create class types
    if(P['HypP']['DatNormMethod']=='Given'):
        dat_norm_wts=P['DatNormCoef']
    else:
        dat_norm_wts={}
        for dt in P['DataTypes']:
            dat_norm_wts[dt] = 1
    P['dat_norm_wts'] = dat_norm_wts
       
    dat_opt_wts={}
    for dt in P['DataTypes']:
        dat_opt_wts[dt] = 1./len(P['DataTypes'])
    P['dat_opt_wts'] = dat_opt_wts
        
                
    creator.create("FitnessMin", base.Fitness, 
                   weights=(-1.0,-1.0, -1.0, -1.0, -1.0,),
                   dat_norm_wts=dat_norm_wts, dat_opt_wts=dat_opt_wts,
                   method = P['HypP']['OptimMethod'],
                   errNorm = P['HypP']['ErrorNorm'],
                   datatypes = P['DataTypes'])
    
    creator.create("Individual", list, fitness=creator.FitnessMin, LocalError ={})

    if(P['HypP']['DatNormMethod']=='Given'):
        P['DatNormCoef'] = P['HypP']['DatNormCoef']

    data_wts = np.ones((len(P['DataTypes']),1))*(1./len(P['DataTypes']))
    P['data_wts'] = {}
    for i in range(len(P['DataTypes'])):
        dt = P['DataTypes'][i]
        P['data_wts'][dt]=data_wts[i]

    toolbox = base.Toolbox()
    register_functions(P, toolbox)
    GI.register_sim_functions(P['HypP'], toolbox)
    P['toolbox']=toolbox

    NGEN = int(P['HypP']['ngen'])
    MU = int(myround(P['HypP']['npop'],4))
    CXPB = 0.9 
    folder=P['folder']
    verbose=P['HypP']['verbose']
    vNames=(P['ModelParamTable']['EventName']+'_'+P['ModelParamTable']['Prop']).values
    ################################################    
    # Algorithm starts here  
    # Basic algorithm structure
    #     Initialize population
    #     for gen i in n_gen:     
    #        i. Evaluate the error of each individual element in the population
    #        ii. Selection of mating pool - NSGA way
    #        iii. Mating to form offspring
    #        iv. Mutating
    # 
    ###############################################    
    P['iterationNum'] = -1

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    toolbox.EvalPop(pop, P, toolbox, folder, verbose, P['ModelParamTable'],
                    vNames)
    
    GA_Util.VizBestInd(pop, folder, P, P['ModelParamTable'], gen=0)

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = emo.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        toolbox.EvalPop(offspring, P, toolbox, folder, verbose, P['ModelParamTable'],
                    vNames)
            
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)

        GA_Util.VizBestInd(pop, folder, P, P['ModelParamTable'], gen=gen)
            
        breakEarly = GI.CheckEarlyStop(P)
        if(breakEarly==1):
            break

    GI.SaveResults2Files(P)

    return 1

def myround(x, base=5):
    return base * round(x/base)

def register_functions(P, toolbox):
    # Create a toolbox of functions
    #   This way, the function always has the same name, but can have 
    #   different content. Allows to keep the same algorithm structure, but also
    #   test some interesting variations.

    std = P['ModelParamTable']['std']
    maxVList = P['ModelParamTable']['maxV'].values
    minVList = P['ModelParamTable']['minV'].values

    # Create the population class
    toolbox.register("SampleValues", sample.SampleParametersMinMax, P)
    toolbox.register("individual", init.initIterate, creator.Individual,
                     toolbox.SampleValues)
    toolbox.register("population", init.initRepeat, list, toolbox.individual)

        
    # i. Evaluation
    if(P['HypP']['ErrorType']=='Global'):
        toolbox.register("EvalPop", GA_Util.EvalPopGlobal)
    else:
        toolbox.register("EvalPop", GA_Util.EvalPopLocal)

    # ii. Selection
    toolbox.register("select", emo.selNSGA2)

    # iii. Mating
    if(P['HypP']['ErrorType']=='Global'):
        if(P['HypP']['MatingMethodGlobal']=='cxTwoPoint'):
            toolbox.register("mate", GA_Util.cxTwoPoint)
        elif(P['HypP']['MatingMethodGlobal']=='cxOnePoint'):
            toolbox.register("mate", GA_Util.cxOnePoint)
        elif(P['HypP']['MatingMethodGlobal']=='cxUniform'):
            toolbox.register("mate", GA_Util.cxUniform, indpb=P['HypP']['MatingSwapProb'])
        elif(P['HypP']['MatingMethodGlobal'] =='cxBlend'):
            toolbox.register("mate", GA_Util.cxBlend, alpha=P['HypP']['MatingSwapRange'])
    elif(P['HypP']['ErrorType']=='Local'):
        if(P['HypP']['MatingMethodLocal'] =='cxLocalBlend'):
            toolbox.register("mate", GA_Util.cxLocalBlend, alpha=P['HypP']['MatingSwapRange'])
        elif(P['HypP']['MatingMethodLocal'] =='cxOnePointLocal'):
            toolbox.register("mate", GA_Util.cxOnePointLocal)        
        elif(P['HypP']['MatingMethodLocal'] =='cxTwoPointLocal'):
            toolbox.register("mate", GA_Util.cxTwoPointLocal)        
        elif(P['HypP']['MatingMethodLocal'] =='cxLocalErrorPropExchange'):
            toolbox.register("mate", GA_Util.cxLocalErrorPropExchange)        
        elif(P['HypP']['MatingMethodLocal'] =='cxLocalBlendExp'):
            toolbox.register("mate", GA_Util.cxLocalBlendExp, alpha=P['HypP']['MatingSwapRange'])   
        
    # iv. Mutating
    if(P['HypP']['MutatingMethod']=='mutGaussian'):
        toolbox.register("mutate", GA_Util.mutGaussianNoddy, mu=0, sigma=std, indpb=P['HypP']['PbMutateParameter'], 
                     maxVList = maxVList, minVList=minVList)
    elif(P['HypP']['MutatingMethod']=='mutUniformFloat'):
        toolbox.register("mutate", GA_Util.mutUniformFloat, indpb=P['HypP']['PbMutateParameter'], 
                     low = list(minVList), up=list(maxVList))
    elif(P['HypP']['MutatingMethod']=='mutPolynomialBounded'):
        toolbox.register("mutate", GA_Util.mutPolynomialBounded, low=minVList, up=maxVList, eta=P['HypP']['Eta'], indpb=P['HypP']['PbMutateParameter'])
    
if __name__ == "__main__":
    tasks = pd.read_pickle('Combo_Scratch/parameters.pkl')
    params = tasks.iloc[6, :]
    params = params.to_dict()
#    params['npop'] = 3
#    params['ngen'] = 3
    params['Windows'] = True
 
    NSGA2_Noddy(params)