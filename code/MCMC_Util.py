# -*- coding: utf-8 -*-
"""
Date: September 14, 2020

@author: ahinoamp@gmail.com

These functions assist in running an MCMC workflow.

Worfklow:

    Repeat until converge
    1. sample values (SamplingHisFileUtil.py)
    2. simulate (SimulationUtilities.py)
    3. calculate error (SimulationUtilities.py)
    4. calc acceptance probability (this file)
    5. accept/reject (this file)
    6. check sufficient convergence (this file)

"""
import SimulationUtilities as sim
import SamplingHisFileUtil as sample
import GeneralInversionUtil as GI
import VisualizationUtilities as Viz
import numpy as np

 
def check_acceptance_global(P):
    '''Check whether a proposed step is accepted when the error type is global.
    1. calculate the error of the last accepted step
    2. calculate the difference between the current and last accepted error
    3. calculate an acceptance probability
    4. draw a random number and determine whether the new step is accepted
    '''
   
    # 1. Calculate the error of the last accepted step using the new data weights
    # Recalculate the mismatch because the weights of different data types can
    # change between 
    lastacceptederror = GI.get_combo_err(P, idx=P['lastAcceptedIdx'],
                                           errNorm = P['HypP']['ErrorNorm'],
                                           datawts = P['data_wts'])
    
    current_err = GI.get_combo_err(P, idx=-1,
                                           errNorm = P['HypP']['ErrorNorm'],
                                           datawts = P['data_wts'])
    # 2. Calculate the difference
    diff = current_err-lastacceptederror      
    
    # 3. Calculate acceptance probability: e^(diff/normalizing_factor)
    normalizingFactor = get_normalizing_factor(P, diff)
    accept, acceptanceProbability = calcAcceptanceProb(diff, normalizingFactor) 
    P['lastAcceptanceProbability'] = acceptanceProbability
    P['lastNormFactor'] = normalizingFactor
    if(P['verbose']):
        print('acceptanceProbability: ' + str(acceptanceProbability))        
        print('Accepted percentage: ' + str(np.mean(P['AllAcceptedList'])))
        print('Last Combo Error: ' + str(current_err))

    if(accept==1):
        P['lastAcceptedIdx'] = P['iterationNum']
        P['AllAcceptedList'].append(1)
    else:
        P['AllAcceptedList'].append(0)
    
def get_normalizing_factor(P, diff):
    '''Calculate the normalizing factor used in the formulate for acceptance
    probability. A high normalizing factor will lead to higher acceptance
    rates (more exploration) and a lower normalizing factor will lead to lower
    acceptance rates (more exploitation)'''
    
    acceptanceList =  P['AllAcceptedList']

    if(P['HypP']['AcceptProbType']=='Track Acceptance'):
        if('lastNormFactor' in list(P.keys())):
            lastNormFactor = P['lastNormFactor']
        else:
            lastNormFactor = np.abs(diff)
            
        nLastTimesConsider = np.min([len(acceptanceList), 20])
        lastAcceptanceRate = np.mean(acceptanceList[len(acceptanceList)-nLastTimesConsider:])
        if(lastAcceptanceRate>=P['HypP']['AcceptanceGoal']):
            #since the acceptance rate is too high, need to decrease normalizaing constant
            normalizingFactor = lastNormFactor*0.95
        else:
            #since the acceptance rate is too low, need to increase normalizaing constant
            normalizingFactor = lastNormFactor*1.05
    elif(P['HypP']['AcceptProbType']=='Annealing'):
        normalizingFactor = P['HypP']['InitialTemperature']*P['HypP']['ReductionRate']**P['iterationNum']
    elif(P['HypP']['AcceptProbType']=='Const diff'):
        normalizingFactor = P['HypP']['ConstNormFactor']
    elif(P['HypP']['AcceptProbType']=='Error must decrease'):
        normalizingFactor = 0.000001
    else:
        normalizingFactor=1
    
    return normalizingFactor

def calcAcceptanceProb(diff, normalizingFactor):
    '''Calculate the acceptance proability and determine proposal acceptance'''

    randN = np.random.rand()
    acceptanceProbability = np.exp(-diff/normalizingFactor)
    if(acceptanceProbability > randN):
        accept=1
    else:
        accept=0

    return accept, acceptanceProbability

def check_acceptance_local(P):
    '''Check whether a proposed step is accepted when the error type is local
    1. calculate the error of the last accepted step
    2. calculate the difference between the current and last accepted error
    3. calculate an acceptance probability
    4. draw a random number and determine whether the new step is accepted
    '''
    nP = P['nParam']

    # 1. Calculate the last error
    lastacceptederror = GI.get_combo_param_err_indices(P, P['lastAcceptedIdx'],
                                                         errNorm = P['HypP']['ErrorNorm'], 
                                                         datawts=P['data_wts'])


    # 2. Calculate the difference
    current_err = GI.get_combo_param_err_idx(P, -1, 
                                                 errNorm = P['HypP']['ErrorNorm'], 
                                                 datawts=P['data_wts']) 
      
    Diff = current_err - lastacceptederror

    # 3. Calculate accept probability 
    normalizingFactor = get_norm_factor_local(P, Diff)     
    P['lastNormFactor'] = normalizingFactor
    acceptanceProbability = np.exp(-Diff/normalizingFactor.reshape((-1,1)))

    # 4. Draw a random number and accept/reject
    randN = np.random.rand(nP,1)
    filterA = acceptanceProbability > randN        
    currentAcceptIdx = np.zeros((nP,), dtype=int)
    currentAcceptIdx[filterA.reshape((-1,))] = P['iterationNum']
    
    # 5. book-keeping    
    lastAcceptedIdx = np.max([P['lastAcceptedIdx'], currentAcceptIdx], axis=0)
    P['lastAcceptedIdx']=lastAcceptedIdx

    currentAcceptReject = np.zeros((nP,), dtype=int)
    currentAcceptReject[filterA.reshape((-1,))] = 1                      
    P['AllAcceptedMatrix'] = np.hstack((P['AllAcceptedMatrix'], currentAcceptReject.reshape((-1,1))))
    P['AllAcceptedList'].append(np.mean(currentAcceptReject))

    if(P['verbose']):          
        print('Accepted percentage: ' + str(np.mean(P['AllAcceptedMatrix'][:])))
        print('Last Combo Error: ' + str(np.mean(P['MismatchList'])))

def get_norm_factor_local(P, diff):
    '''Get local error normalization parameter for acceptance function'''

    if(P['HypP']['AcceptProbType']=='Error must decrease'):
        normalizingFactor=0.000001*np.ones((P['nParam'],))
    elif(P['HypP']['AcceptProbType']=='Const diff'):
        normalizingFactor = P['ConstNormFactor']*np.ones((P['nParam'],))
    elif(P['HypP']['AcceptProbType']=='Annealing'):
        normalizingFactor = P['InitialTemperature']*P['ReductionRate']**P['iterationNum']
        normalizingFactor = normalizingFactor*np.ones((P['nParam'],))
    elif(P['HypP']['AcceptProbType']=='Track Acceptance'):
        nLastTimesConsider = np.min([np.shape(P['AllAcceptedMatrix'])[1], 20])
        lastAcceptanceRate = np.mean(P['AllAcceptedMatrix'][:,np.shape(P['AllAcceptedMatrix'])[1]-nLastTimesConsider:], axis=1)
        if('lastNormFactor' in list(P.keys())):
            normalizingFactor = P['lastNormFactor']
        else:
            normalizingFactor = np.abs(diff)
        #since the acceptance rate is too high, need to decrease normalizaing constant
        filterHigh = lastAcceptanceRate>=P['HypP']['AcceptanceGoal']
        normalizingFactor[filterHigh] = normalizingFactor[filterHigh]*0.95
        #since the acceptance rate is too low, need to increase normalizaing constant
        filterLow = lastAcceptanceRate<P['HypP']['AcceptanceGoal']
        normalizingFactor[filterLow] = normalizingFactor[filterLow]*1.05

    return normalizingFactor

def updateDataTypeWeights(P):
    '''change the weights assigned to each datatype.
    This ability was created to try and respect more of the data types by 
    changing the emphasis on the different data types'''
    
    nDataType = len(P['HypP']['DataTypes'])
    if(P['HypP']['MO_WeightingMethod']=='Proportions'):
        sample5 = np.random.uniform(0,1,nDataType)
        weights = sample5/np.sum(sample5)
    elif(P['HypP']['MO_WeightingMethod']=='Extreme'):
        index = int(np.mod(np.floor(float(P['iterationNum'])/P['HypP']['MCMC_SwitchWeightFreq'])+P['StartIdx'], nDataType))
        weights = np.zeros((nDataType,))
        weights[index] =1
    else:
        weights = np.ones((nDataType,))/float(nDataType)
    
    P['data_wts'] = {}
    for i in range(nDataType):
        dt = P['DataTypes'][i]
        P['data_wts'][dt]=weights[i]
    
def AcceptFirstRound(P):
    '''The first MCMC round is always accepted'''

    P['ParamHis'] = P['ModelParamTable']
    P['AllAcceptedList'] = [1]
  
    if(P['HypP']['ErrorType']=='Global'):
        P['lastAcceptedIdx']  = 0
        P['lastAcceptanceProbability'] = 1
    else:
        P['lastAcceptedIdx']  = np.zeros((P['nParam'],), dtype=int)
        P['AllAcceptedMatrix'] = np.ones((P['nParam'],1))
        
def explore_space(P):
    """Several iterations of exploration.
    Run a number of iterations where the parameter values are sampled randomly
    between the ranges of the minimum and maximum values for the parameter.
    """
    for i in range(int(P['HypP']['nExploreRuns'])):
        P['iterationNum'] = P['iterationNum']+1

        if(P['verbose']):
            print('Run Number: ' + str(P['iterationNum']))

        # 1. sample model values
        parameters = sample.SampleParametersMinMax(P)
        # 2. write parameter values to file
        sample.UpdateTemplateGivenParam(parameters, P['ModelParamTable'], 
                                        P['SampledInputFileName'], P)

        # 3. book keep
        P['ModelParamTable'][str(P['iterationNum'])] = parameters
        P['FullParamHis'].loc[P['OptimizeParametersIdx'],str(P['iterationNum'])] = parameters
        P['FullParamHis'].loc[~P['OptimizeParametersIdx'],str(P['iterationNum'])] = P['FullParamHis'].loc[~P['OptimizeParametersIdx'],'minV']


        sim.simulate_calc_mismatch(P)

        # Update local Weights if necessary
        if((P['HypP']['ErrorType'] == 'Local') & (P['HypP']['LocalWeightsMode'] == 'Many')):
            GI.UpdateLocalWeights(P)

        AcceptExplorationRounds(P)

        if(np.mod((P['iterationNum']),P['HypP']['OutputImageFreq'])==0):
            GI.OutputImageAndHisFile(P)


    #Calculate normalization factor if necessary
    if(P['HypP']['DatNormMethod']=='MedianInitialRounds'):
        GI.CalcDatNormCoef(P)    

    #Choose the best point to start your round    
    StartChainFromBestOne(P)


def AcceptExplorationRounds(P):
    '''Exploration rounds are always accepted'''

    P['AllAcceptedList'].append(1)

    if(P['HypP']['ErrorType']=='Global'):
        P['lastAcceptedIdx'] = P['iterationNum']
    else:
        currentAcceptReject = np.ones((P['nParam'],), dtype=int)       
        P['AllAcceptedMatrix'] = np.hstack((P['AllAcceptedMatrix'], currentAcceptReject.reshape((-1,1))))
        currentAcceptIdx = np.ones((P['nParam'],), dtype=int)*P['iterationNum']
        P['lastAcceptedIdx'] = currentAcceptIdx

    if(P['verbose']):
        print('Accepted percentage: ' + str(np.mean(P['AllAcceptedList'])))
   
def StartChainFromBestOne(P):
    '''Continue the MCMC chain from the best model found in the exploration 
    round'''
    
    if(P['HypP']['ErrorType']=='Global'):
        idxBest = np.argmin(GI.get_combo_err_list(P))
        P['lastAcceptedIdx']  = idxBest
    else:
        errorMatrix = GI.get_combo_param_err_matrix(P)
        idxBest = np.argmin(errorMatrix, axis=1)
        P['lastAcceptedIdx']  = idxBest
        
