# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:55:26 2020

@author: ahinoamp

Send many optimisation threads to different CPUs and vary the hyper parameters
according to different distributions.
"""

import pandas as pd
import random   
import multiprocessing
import numpy as np
import sys
import OptimisationSwitchBoard as switchboard

def getHypPara():
    
    #############################
    ## MCMC
    #############################
 
    HyperParameters = []

    param = {}
    param['Type']= 'pRandChoice'
    param['parameters']= [10,11, 12, 13, 14, 15, 16, 17, 18, 19]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    param['Name']='ScenarioNum'
    HyperParameters.append(param)
   
    param = {}
    param['Name']='SimulationShiftType'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Median Datum Shift', 'Median Const Datum Shift']
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)
    
    param = {}
    param['Type']= 'pRandChoice'
    param['parameters']= ['MedianInitialRounds', 'Given']
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    param['Name']='DatNormMethod'
    HyperParameters.append(param)

    param = {}
    param['Name']='ExplorationRate'
    param['Type']= 'pRandChoice'
    param['parameters']= ['LinearErrorBased', 'None']
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='ErrorType'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Global', 'Local']
    param['Methods']= ['MCMC', 'GA', 'NSGA', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='LocalWeightsMode'
    param['Type']= 'pRandChoice'
    param['Condition']= [['ErrorType', 'Local']]
    param['parameters']= ['Once', 'Many']
    param['Methods']= ['MCMC', 'NSGA', 'GA', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='ErrorNorm'
    param['Type']= 'pRandChoice'
    param['parameters']= ['L1', 'L2', 'Lhalf']
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='ExplorationStage'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Explore']
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='AcceptProbType'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Track Acceptance', 'Error must decrease', 'Const diff']
    param['Methods']= ['MCMC']
    HyperParameters.append(param)

    param = {}
    param['Name']='AcceptProbType'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Annealing']
    param['Methods']= ['Annealing']
    HyperParameters.append(param)
    
    param = {}
    param['Name']='cubesize'
    param['Type']= 'pRandChoice'
    param['parameters']= [100, 150]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='AcceptanceGoal'
    param['Type']= 'pRandFloat'
    param['Condition']= [['AcceptProbType', 'Track Acceptance']]
    param['parameters']= [0.05, 0.20]
    param['Methods']= ['MCMC']
    HyperParameters.append(param)

    param = {}
    param['Name']='ConstNormFactor'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.001, 0.015]
    param['Methods']= ['MCMC']
    HyperParameters.append(param)

    param = {}
    param['Name']='InitialTemperature'
    param['Type']= 'pRandFloat'
    param['Condition']= [['AcceptProbType', 'Annealing']]
    param['parameters']= [0.001, 0.025]
    param['Methods']= ['Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='ReductionRate'
    param['Type']= 'pRandFloat'
    param['Condition']= [['AcceptProbType', 'Annealing']]
    param['parameters']= [0.95, 0.999]
    param['Methods']= ['Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='GlobalMoveEachDir'
    param['Type']= 'pRandFloat'
    param['parameters']= [200, 900]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='XYZ_Axes_StepStd'
    param['Type']= 'pRandFloat'
    param['parameters']= [70, 150]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='Dip_StepStd'
    param['Type']= 'pRandFloat'
    param['parameters']= [2, 7]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='Slip_StepStd'
    param['Type']= 'pRandFloat'
    param['parameters']= [30, 90]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='localWeightRadiusMult'
    param['Type']= 'pRandFloat'
    param['parameters']= [1, 2.5]
    param['Condition']= [['ErrorType', 'Local']]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='DipDirection_StepStd'
    param['Type']= 'pRandFloat'
    param['parameters']= [2, 10]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)
    
    param = {}
    param['Name']='SlipParam'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.09, 0.2]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='SteppingSizeMult'
    param['Type']= 'pRandFloat'
    param['parameters']= [1.0/1.1, 1.0/0.7]
    param['Condition']= [['ExplorationRate', 'LinearErrorBased']]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)
      
    param = {}
    param['Name']='MaxFaultMarkerError'
    param['Type']= 'pRandFloat'
    param['parameters']= [525, 526]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='AmplitudeRatioChange'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.05, 0.25]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='AzimuthMoveEachDirection'
    param['Type']= 'pRandFloat'
    param['parameters']= [3, 10]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='AxisRatioChange'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.02, 0.25]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='DipMoveEachDirection'
    param['Type']= 'pRandFloat'
    param['parameters']= [25, 40]
    param['Methods']= ['MCMC', 'Annealing', 'NSGA', 'GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='StartUpdateLocalWeight'
    param['Type']= 'pRandInt'
    param['parameters']= [35, 60]
    param['Condition']= [['ErrorType', 'Local']]
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='UpdateLocalWeightsFreqRuleBased'
    param['Type']= 'pRandInt'
    param['parameters']= [35, 70]
    param['Condition']= [['LocalWeightsMode', 'Many']]
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='nExploreRuns'
    param['Type']= 'pRandInt'
    param['parameters']= [40, 70]
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    #############################
    ## GA
    #############################
    
    param = {}
    param['Name']='MO_WeightingMethod'
    param['Type']= 'pRandChoice'
    param['parameters']= ['Proportions', 'Extreme', 'Equal']
    param['Methods']= ['GA', 'MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='SelectionMethod'
    param['Type']= 'pRandChoice'
    param['parameters']= ['selTournament', 'selStochasticUniversalSampling', 'selRoulette']
    param['Methods']= ['GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MatingMethodGlobal'
    param['Type']= 'pRandChoice'
    param['parameters']= ['cxTwoPoint','cxOnePoint','cxUniform']
    param['Condition']= [['ErrorType', 'Global']]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MatingMethodLocal'
    param['Type']= 'pRandChoice'
    param['parameters']= ['cxOnePointLocal','cxTwoPointLocal', 'cxLocalErrorPropExchange']
    param['Condition']= [['ErrorType', 'Local']]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MutatingMethod'
    param['Type']= 'pRandChoice'
    param['parameters']= ['mutPolynomialBounded', 'mutGaussian', 'mutUniformFloat']
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='IndMutatingProb'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.2, 0.4]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='IndMatingProb'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.6, 1]
    param['Methods']= ['GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='PbMutateParameter'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.1, 0.3]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MatingSwapRange'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.6, 1]
    param['Condition']= [['MatingMethodGlobal', 'cxBlend'], ['MatingMethodLocal', 'cxLocalBlendExp'],
                         ['MatingMethodLocal', 'cxLocalBlend']]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MatingSwapProb'
    param['Type']= 'pRandFloat'
    param['parameters']= [0.3, 0.7]
    param['Condition']= [['MatingMethodGlobal', 'cxUniform']]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='Eta'
    param['Type']= 'pRandFloat'
    param['parameters']= [80, 120]
    param['Condition']= [['MutatingMethod', 'mutPolynomialBounded']]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)
    
    param = {}
    param['Name']='npop'
    param['Type']= 'pRandInt'
    param['parameters']= [25, 80]
    param['Methods']= ['GA', 'NSGA']
    HyperParameters.append(param)

    param = {}
    param['Name']='nruns'
    param['Type']= 'pRandInt'
    param['parameters']= [2000, 4000]
    param['Methods']= ['GA', 'NSGA', 'MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='MO_SwitchWeightFreq'
    param['Type']= 'pRandInt'
    param['parameters']= [1, 6]
    param['Condition']= [['MO_WeightingMethod', 'Extreme']]
    param['Methods']= ['GA']
    HyperParameters.append(param)

    param = {}
    param['Name']='MCMC_SwitchWeightFreq'
    param['Type']= 'pRandInt'
    param['parameters']= [1, 30]
    param['Condition']= [['MO_WeightingMethod', 'Extreme']]
    param['Methods']= ['MCMC', 'Annealing']
    HyperParameters.append(param)

    param = {}
    param['Name']='TournamentSize'
    param['Type']= 'pRandInt'
    param['parameters']= [4, 12]
    param['Condition']= [['SelectionMethod', 'selTournament']]
    param['Methods']= ['GA']
    HyperParameters.append(param)
   
    return HyperParameters

def getAllKeys(HypPara):
    
    AllKeys = []
    for i in range(len(HypPara)):
        AllKeys.append(HypPara[i]['Name'])
                       
    return np.unique(AllKeys)

def sampleParameters(HypPara, params):
 
    optimMethod = params['OptimMethod']
    for i in range(len(HypPara)):
        HP_i = HypPara[i]

        if(optimMethod not in HP_i['Methods']):
            continue

        genType = HP_i['Type']
        name = HP_i['Name']

        #check if the parameter is necessary
        if('Condition' in HP_i.keys()):
            MeetsCondition = 0
            for c in range(len(HP_i['Condition'])):
                condition_i = HP_i['Condition'][c]
                if(condition_i[0] not in params.keys()):
                    continue
                if(params[condition_i[0]]==condition_i[1]):
                    MeetsCondition=MeetsCondition+1
            if(MeetsCondition==0):
                continue

        if genType == 'pRandChoice':
            params[name] = random.choice(HP_i['parameters'])
        if genType == 'pRandFloat':
            params[name] = random.uniform(HP_i['parameters'][0],HP_i['parameters'][1])                   
        if genType == 'pRandInt':
            params[name] = int(random.randint(HP_i['parameters'][0],HP_i['parameters'][1])) 
    

    if(optimMethod=='GA'):
        params['ngen']=int(float(params['nruns'])/float(params['npop']))

        if(params['ErrorType']=='Local'):
            params['LocalWeightsMethod'] = 'RuleBased'

        if('TournamentSize' in params.keys()):
            params['TournamentSize'] = int(np.min([params['npop']/4, params['TournamentSize']]))

    if(optimMethod=='NSGA'):
        params['ngen']=int(float(params['nruns'])/float(params['npop']))
        if(params['ErrorType']=='Local'):
            params['LocalWeightsMethod'] = 'RuleBased'

    
class Predictor(multiprocessing.Process):
    def __init__(self, input_queue, output_queue, gpu_id):
        multiprocessing.Process.__init__(self)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.gpu_id = gpu_id

    def initialize_logging(self, num):
        sys.stdout = open('OutputFiles/'+str(num) + ".out", "a")
        sys.stderr = open('OutputFiles/'+str(num) + "err.out", "a")

    def run(self):
        while True:
            input = self.input_queue.get()
            if input is None:
                self.input_queue.task_done()
                self.output_queue.put("Finished with processor %d" % self.gpu_id)
                break
            else:
                try:
#                    self.initialize_logging(input['thread_num'])
                    switchboard.OptimisationSwitchBoard(input)
                    self.input_queue.task_done()
                    self.output_queue.put('complete- run #'+str(input['thread_num'])+' on cpu#'+str(self.gpu_id))
                except Exception:
#                    traceback.print_exc(file='exceptionfile.txt')
                    self.input_queue.task_done()
                    self.output_queue.put('There was an exception- run #'+str(input['thread_num'])+'on cpu#'+str(self.gpu_id))
        return

if __name__ == "__main__":
      
  
    #sample the realizations
    num_threads =  25
    OutputImageFreq = 15
    tasks = []
    num_gpus = 16
    
    HypPara = getHypPara()
    
    for i in range(num_threads):
        params = {}
        
        optimMethods = ['MCMC', 'GA', 'NSGA', 'Annealing']
#        optimMethod = np.random.choice(optimMethods)
        optimMethod = optimMethods[np.mod(i, 4)]
        
        params['OptimMethod'] =optimMethod
        sampleParameters(HypPara, params)        
            
        keys = list(params.keys())
        allPossibleKeys = getAllKeys(HypPara)
        for key_i in allPossibleKeys:
            if(key_i not in keys):
                params[key_i]=np.nan

       
        params['thread_num'] = i        
        params['OutputImageFreq'] = OutputImageFreq        
        params['Toy']=False
        params['verbose']=False
        params['BaseFolder']='Combo_Scratch'
        params['GeneralPerturbStrategy']='OnlyGlobal'
        params['ControlPointMovementUnits'] = 'Absolute value'
        params['errCalcMethodFaultMarker']= 'Distance'
        params['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultMarkers']
        params['JoinType']='LINES'
        params['xy_origin']=[316448, 4379166, -2700]
        params['xy_extent'] = [8850, 9000,3900]
        params['DatNormCoef'] = {'Grav': 2.4, 'Tracer': 1.0, 
                                'FaultMarkers': 500, 'GT': 315, 'Mag':300}
        params['verbose']=True    
        params['graniteIdx'] = 4
        params['Windows'] = False
        params['jupyter'] = False
        tasks.append(params)

    HypParametersPD = pd.DataFrame(tasks)    
    HypParametersPD.to_csv('Combo_Scratch/parameters.csv')
    HypParametersPD.to_pickle('Combo_Scratch/parameters.pkl')
    
    p_list = []
    input_queue = multiprocessing.JoinableQueue()
    output_queue = multiprocessing.Queue()
    for i in range(num_gpus):
        p = Predictor(input_queue, output_queue, i)
        p_list.append(p)

    for task in tasks:
        input_queue.put((task))

    for p in p_list:
        p.start()

    for i in range(num_gpus):
        input_queue.put(None)

    for i in range(len(tasks)+num_gpus):
        print(output_queue.get())

    input_queue.join()

    for p in p_list:
        p.join()