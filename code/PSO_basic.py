# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:58:58 2020

@author: ahinoamp
"""

import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
import PSO_GA_Utilities as PSOGA
import PSO_GA_Utilities as PSOGA
import random
import LoadInputDataUtility as DI
from collections.abc import Sequence
import math
import random
import pandas as pd
from itertools import repeat
from pathlib import Path

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def generate(size, smin, smax, ModelParametersTable):
    
    part = creator.Particle(PSOGA.SampleValues(ModelParametersTable)) 
    part.speed = [random.uniform(smin[i], smax[i]) for i in range(size)]
    part.smin = smin
    part.smax = smax
    return part

def updateParticle(part, best, phi1, phi2, vmin, vmax):
    u1 = []
    u2 = []
    for i in range(len(part)):
        u1.append(random.uniform(0, phi1))
        u2.append(random.uniform(0, phi2))
    vec_personal_best = np.asarray(u1)*(np.asarray(part.best) - np.asarray(part))
    vec_global_best = np.asarray(u2)*(np.asarray(best) - np.asarray(part))
    TotalSpeed = part.speed + vec_personal_best+vec_global_best 
    for i, speed in enumerate(TotalSpeed):
        if abs(speed) < part.smin[i]:
            TotalSpeed[i] = math.copysign(part.smin[i], speed)
        elif abs(speed) > part.smax[i]:
            TotalSpeed[i] = math.copysign(part.smax[i], speed)
    NewLocation = np.asarray(part)+TotalSpeed
    NewLocationChecked = []
    for i in range(len(part)):
        val = NewLocation[i]
        if(val<vmin[i]):
            val = vmin[i]
        if(val>vmax[i]):
            val = vmax[i]
        NewLocationChecked.append(val)
    part[:] = NewLocationChecked
        


def Basic_PSO_Noddy(P):
    folder = P['BaseFolder']+'/Thread'+str(P['thread_num'])+'/'
    P['folder']=folder
    Path(folder).mkdir(parents=True, exist_ok=True)
    folderHis = folder+'HistoryFileInspection/'
    Path(folderHis).mkdir(parents=True, exist_ok=True)
    folderViz = folder+'VisualInspection/'
    Path(folderViz).mkdir(parents=True, exist_ok=True)
    folderGrav = folder+'GravityRealization/'
    Path(folderGrav).mkdir(parents=True, exist_ok=True)
    logfile = open(folder+'logfile.txt',"w") 

    P['SampledInputFileName'] = folder+'scenario_scratch.his'

    P['CalcGravity']=True
    P['CalcMagnetics']=True
    P['CalcGraniteTop']=True
    P['CalcTracer']=True
    P['CalcFaultIntersection']=True
    P['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultIntersection']
    P['JoinType']='LINES'

    #############################
    ## 0 define the model grid boundaries + some other input stuff
    #############################  
    P['xy_origin']=[317883,4379246, 1200-4000]
    P['xy_extent'] = [9000,9400,4000]
    
    P['output_name'] = folder+'noddy_out'
    P['iterationNum'] =0
    
    P['verbose']=False    
    P['Grav_comboNormalizingFactor'] = 2.37
    P['Mag_comboNormalizingFactor'] = 3100
    P['Tracer_comboNormalizingFactor'] = 0.8125
    P['GT_comboNormalizingFactor'] = 275
    P['FaultIntersection_comboNormalizingFactor'] = 450    
    
    DI.loadData(P)

    
    ModelParametersTable = PSOGA.InitializeParameters(P)
    std = ModelParametersTable['proposal_std_or_range'].values
    maxVList = ModelParametersTable['maxV'].values
    minVList = ModelParametersTable['minV'].values
    vNames=(ModelParametersTable['EventName']+'_'+ModelParametersTable['Property']).values
    nParameters = len(ModelParametersTable)
    #def Evaluate(parameters):  
    #    return error
    

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, 
        smin=None, smax=None, best=None)


   
    #Structure initializers
    #An individual is a list that represents the position of each queen.
    #Only the line is stored, the column is the index of the number in the list.
    toolbox = base.Toolbox()
    toolbox.register("SampleValues", PSOGA.SampleValues, ModelParametersTable)
    toolbox.register("particle", generate, size=nParameters, smin=0.15*std, smax=2*std, 
                     ModelParametersTable=ModelParametersTable)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=P['WeightPersonalBest'], phi2=P['WeightGlobalBest'], 
                     vmin = minVList, vmax= maxVList)
    toolbox.register("evaluate", PSOGA.evalNoddyUniObjPSO, P=P, ModelParametersTable=ModelParametersTable)

    pop = toolbox.population(n=P['npop'])
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None

    for g in range(int(P['ngen'])):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
        winningParameters=np.asarray(best)
        PSOGA.parametersViz(winningParameters, folder+'Viz_'+str(0), P, ModelParametersTable, g)

    gen = logbook.select("gen")
    mins = logbook.select("Min")
    maxs = logbook.select("Max")
    avg = logbook.select("Avg")
        
    Results = pd.DataFrame({'gen': gen, 'Mismatch': mins, 'maxs': maxs, 'avg': avg})
    Results.to_csv(folder+'Mismatch.csv')
    
    return pop, logbook, best

if __name__ == "__main__":
    tasks = pd.read_csv('PSO_GA_Scratch/parameters.csv')
    params = tasks.iloc[6, :]
    params = params.to_dict()
    params['ngen'] = 2
    params['npop'] = 10
    params['MatingSwapRange'] = 0.5
    params['MatingMethod']='cxBlend'
    params['SelectionMethod']='selStochasticUniversalSampling'
    params['Eta']=100
    params['MutatingMethod']='mutUniformFloat'
    params['mutpb']=1
    params['WeightPersonalBest']=2
    params['WeightGlobalBest']=1
    pop, logbook, best = Basic_PSO_Noddy(params)