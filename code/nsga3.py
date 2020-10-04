# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:30:36 2020

@author: ahinoamp
"""

from math import factorial
import random

import matplotlib.pyplot as plt
import numpy
import pymop.factory

from deap import algorithms
from deap import base
from deap.benchmarks.tools import igd
from deap import creator
from deap import tools

import PSO_GA_Utilities as PSOGA
import random
import LoadInputDataUtility as DI
from collections.abc import Sequence
import math
import random

from itertools import repeat

import numpy as np

def mutGaussianNoddy(individual, mu, sigma, indpb, minVList, maxVList):
    """This function applies a gaussian mutation of mean *mu* and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.

    :param individual: Individual to be mutated.
    :param mu: Mean or :term:`python:sequence` of means for the
               gaussian addition mutation.
    :param sigma: Standard deviation or :term:`python:sequence` of
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.

    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)

    for i, m, s, maxV, minV in zip(range(size), [mu]*size, sigma, maxVList, minVList):
        if random.random() < indpb:
            individual[i] += random.gauss(m, s)
            if(individual[i]>maxV):
                individual[i] = maxV
            if(individual[i]<minV):
                individual[i] = minV
    return individual,


def evalNoddy(individual, P=0, ModelParametersTable=0):
    """Evaluation function for 5 data types in Noddy.
    """   
    error = PSOGA.EvaluateError(individual, P, ModelParametersTable)
        
    return error

def SampleValues(ModelParametersTable):
    
    minVList = ModelParametersTable['minV']
    maxVList = ModelParametersTable['maxV']
    parameters = []
    for i in range(len(ModelParametersTable)):
        minV = minVList[i]
        maxV = maxVList[i]
        randomNumber = np.random.rand()
        paramVal = randomNumber*(maxV-minV)+minV
        parameters.append(paramVal)
    return parameters

P = {}
P['ScenarioNum']= 15
P['cubesize']= 150
P['GlobalMoveEachDir'] = 600
P['SlipParam']= 0.08
P['AmplitudeRatioChange'] = 0.18
P['AzimuthMoveEachDirection'] = 10
P['AxisRatioChange']=0.15
P['DipMoveEachDirection'] = 15
folder = 'PSO_GA_Scratch/'
P['SampledInputFileName'] = folder+'scenario_scratch.his'
P['output_name'] = folder+'noddy_out'
P['xy_extent'] = [9000,9400,4000]
P['JoinType']='LINES'
P['XYZ_Axes_StepStd'] = 100
P['Dip_StepStd'] = 4
P['DipDirection_StepStd'] = 12
P['Slip_StepStd'] = 60
P['CalcGravity']=True
P['CalcMagnetics']=True
P['CalcGraniteTop']=True
P['CalcTracer']=True
P['CalcFaultIntersection']=True
P['DataTypes'] = ['Grav', 'Mag', 'Tracer', 'GT', 'FaultIntersection']
P['xy_origin']=[317883,4379246, 1200-4000]
P['SimulationShiftType']='Datum Shift'
P['ErrorNorm']='L1'
P['iterationNum']=0
P['ErrorType']='Global'
P['errCalcMethodFaultIntersection'] = 'Distance'
P['MaxFaultIntersectionError']=500
P['verbose']=False

P['Grav_comboNormalizingFactor'] = 2.37
P['Mag_comboNormalizingFactor'] = 3100
P['Tracer_comboNormalizingFactor'] = 0.8125
P['GT_comboNormalizingFactor'] = 275
P['FaultIntersection_comboNormalizingFactor'] = 450



DI.loadData(P)


ModelParametersTable = PSOGA.InitializeParameters(P)
std = ModelParametersTable['proposal_std_or_range']
maxVList = ModelParametersTable['maxV']
minVList = ModelParametersTable['minV']


# Problem definition
PROBLEM = "dtlz2"
NOBJ = 5
K = 10
NDIM = NOBJ + K - 1
P = len(ModelParametersTable)
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
# BOUND_LOW, BOUND_UP = 0.0, 1.0
problem = pymop.factory.get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
# ##

# # Algorithm parameters
MU = int(H + (4 - H % 4))
NGEN = 400
CXPB = 1.0
MUTPB = 1.0
##

# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)

# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0,-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)
##


# Toolbox initialization
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

toolbox = base.Toolbox()
#toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.SampleValues)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalNoddy, P=P, ModelParametersTable=ModelParametersTable)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=minVList, up=maxVList, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=minVList, up=maxVList, eta=20.0, indpb=0.05)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
##


def main(seed=None):
    random.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean, axis=0)
    stats.register("std", numpy.std, axis=0)
    stats.register("min", numpy.min, axis=0)
    stats.register("max", numpy.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN):
        offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, MU)

        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)


    return pop, logbook


if __name__ == "__main__":
    pop, stats = main()
    pop_fit = numpy.array([ind.fitness.values for ind in pop])

#    pf = problem.pareto_front(ref_points)
  #  print(igd(pop_fit, pf))

    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as Axes3d

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    p = numpy.array([ind.fitness.values for ind in pop])
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], marker="o", s=24, label="Final Population")

  #  ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], marker="x", c="k", s=32, label="Ideal Pareto Front")

 #   ref_points = tools.uniform_reference_points(NOBJ, P)

    ax.scatter(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], marker="o", s=24, label="Reference Points")

    ax.view_init(elev=11, azim=-25)
    ax.autoscale(tight=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("nsga3.png")