# -*- coding: utf-8 -*-
"""
Created on September 16, 2020

@author: ahinoamp@gmail.com 

This file has utilities for running the Genetic algorithms (include NSGA)
"""

import pandas as pd
import GeneralInversionUtil as GI
import VisualizationUtilities as Viz
import PriorUncertaintyUtil as Unc
import LoadInputDataUtility as DI
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from collections.abc import Sequence
from itertools import repeat
import random
import copy
from functools import partial
from operator import attrgetter

import SamplingHisFileUtil as sample
import SimulationUtilities as sim

###########################
## Evaluation utilities
###########################            
def EvaluateError(individual, P):
    '''Evaluate the error for a single individual'''

    P['iterationNum'] = P['iterationNum']+1   
    
    # Sample new values
    sample.UpdateTemplateGivenParam(individual, P['ModelParamTable'], 
                                    P['SampledInputFileName'], P)    

    # book keep
    P['ModelParamTable'][str(P['iterationNum'])] = individual
    P['FullParamHis'].loc[P['OptimizeParametersIdx'],str(P['iterationNum'])] = individual
    P['FullParamHis'].loc[~P['OptimizeParametersIdx'],str(P['iterationNum'])] = P['FullParamHis'].loc[~P['OptimizeParametersIdx'],'minV']

    # Simulate the data and calculate the error associated with the parameters
    sim.simulate_calc_mismatch(P)
             
    Errors = {}
    for dt in P['DataTypes']:
        Errors[dt] = P[dt]['L1MismatchMatrix'][:, -1]
        
    return Errors
            
def EvalPopGlobal(population, P, toolbox, folder, verbose, 
                  ModelParamTable, variableNames):

            
    # Evaluate the individuals with an invalid fitness
    # Keep the information regarding the best individual 
    # (of those that are recalculated)
    lowestErr = 10000000

    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.setValues(EvaluateError(ind, P))
            if(ind.fitness.getValues()[0] < lowestErr):
                lowestErr=ind.fitness.getValues()[0]
                P['BestIdxGen'] = P['iterationNum']
                P['BestIdxP'] = copy.deepcopy(P)

    if((P['iterationNum']<50) & (P['HypP']['DatNormMethod']=='MedianInitialRounds')):
        UpdateDatNormCoef(P, population)
        
def UpdateDatNormCoef(P, population):
    
    GI.CalcDatNormCoef(P)

    for ind in population:
        ind.fitness.setDatNormWts(P['DatNormCoef'])  
            
def EvalPopLocal(population, P, toolbox, folder, verbose,
                 ModelParamTable, variableNames):

    # Evaluate the individuals with an invalid fitness
    # Keep the information regarding the best individual 
    # (of those that are recalculated)
    lowestErr = 10000000

    for ind in population:
        if not ind.fitness.valid:
            ind.fitness.setValues(EvaluateError(ind, P))
            UpdateLocalWts(P)
            SetLocalNonNormErrors(P, ind)
            if(ind.fitness.getValues()[0] < lowestErr):
                lowestErr=ind.fitness.getValues()[0]
                P['BestIdxGen'] = P['iterationNum']
                P['BestIdxP'] = copy.deepcopy(P)

    if((P['iterationNum']<50) & (P['HypP']['DatNormMethod']=='MedianInitialRounds')):
        UpdateDatNormCoef(P, population)    
    
    CalcNormLocalErrs(population, P)

def SetLocalNonNormErrors(P, ind):
    '''calculate the combined local error at specified step indices'''
    
    ind.LocalError = {}

    for dt in P['DataTypes']:
        err = P[dt]['L1MismatchMatrix'][:, -1]
        ind.LocalError[dt]=err

def CalcNormLocalErrs(population, P):

    for ind in population:
        ComboErrorPerParameter = np.zeros((P['nParam'],))
        for dt in P['DataTypes']:
            err = ind.LocalError[dt]
            norm_err = err/P['DatNormCoef'][dt]
            if(P['HypP']['ErrorNorm']=='L2'):
                norm_err = norm_err**2
            elif(P['HypP']['ErrorNorm']=='Lhalf'):
                norm_err = norm_err**0.5           
            local_err = np.dot(norm_err,P[dt]['LocalErrorWeights'])
            weighted_err = local_err*P['dat_opt_wts'][dt]
            ComboErrorPerParameter = ComboErrorPerParameter + weighted_err
        ind.LocalError['Combo']= ComboErrorPerParameter
    
def UpdateLocalWts(P):

    if((P['HypP']['LocalWeightsMode']=='Once') & (P['iterationNum']==0)):
        for dt in P['DataTypes']:      
             P[dt]['LocalErrorWeights'] = GI.CalcLocalErrWts_MinMaxRanges(P, dt)
    elif(P['HypP']['LocalWeightsMode']=='Many'):
        for dt in P['DataTypes']:      
             P[dt]['LocalErrorWeights'] = GI.CalcLocalErrWts(P, dt)            
            
def UpdateOptWeights(population, P, gen):

    weights = None
    if(P['HypP']['MO_WeightingMethod']=='Proportions'):
        sample5 = np.random.uniform(0,1,5)
        weights = sample5/np.sum(sample5)
        for ind in population:
            ind.fitness.setDataOptWts(weights)

    elif(P['HypP']['MO_WeightingMethod']=='Extreme'):
        index = int(np.mod(np.floor(float(gen)/P['HypP']['MO_SwitchWeightFreq'])+P['HypP']['StartIdx'], 5))
        weights = np.zeros((5,))
        weights[index] =1
        for ind in population:
            ind.fitness.setDataOptWts(weights)
    else:
        if(gen==0):
            for ind in population:
                weights = np.ones((5,))/5.0
                ind.fitness.setDataOptWts(weights)
    
    if(weights is not None):
        P['data_wts']={}
        for i in range(len(P['DataTypes'])):
            dt = P['DataTypes'][i]
            P['data_wts'][dt] = weights[i]

###############################################
## Algorithmic structure from deap code
###############################################

def varAnd(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover **and** mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so returned population is
    independent of the input population.

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: A list of varied individuals that are independent of their
              parents.

    The variation goes as follow. First, the parental population
    :math:`P_\mathrm{p}` is duplicated using the :meth:`toolbox.clone` method
    and the result is put into the offspring population :math:`P_\mathrm{o}`.  A
    first loop over :math:`P_\mathrm{o}` is executed to mate pairs of
    consecutive individuals. According to the crossover probability *cxpb*, the
    individuals :math:`\mathbf{x}_i` and :math:`\mathbf{x}_{i+1}` are mated
    using the :meth:`toolbox.mate` method. The resulting children
    :math:`\mathbf{y}_i` and :math:`\mathbf{y}_{i+1}` replace their respective
    parents in :math:`P_\mathrm{o}`. A second loop over the resulting
    :math:`P_\mathrm{o}` is executed to mutate every individual with a
    probability *mutpb*. When an individual is mutated it replaces its not
    mutated version in :math:`P_\mathrm{o}`. The resulting :math:`P_\mathrm{o}`
    is returned.

    This variation is named *And* because of its propensity to apply both
    crossover and mutation on the individuals. Note that both operators are
    not applied systematically, the resulting individuals can be generated from
    crossover only, mutation only, crossover and mutation, and reproduction
    according to the given probabilities. Both probabilities should be in
    :math:`[0, 1]`.
    """
    offspring = [toolbox.clone(ind) for ind in population]

    # Apply crossover and mutation on the offspring
    for i in range(1, len(offspring), 2):
        if random.random() < cxpb:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1],
                                                          offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values

    for i in range(len(offspring)):
        if random.random() < mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring

###########################
## Selection methods
###########################   
def selRandom(individuals, k):
    """Select *k* individuals at random from the input *individuals* with
    replacement. The list returned contains references to the input
    *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the
    python base :mod:`random` module.
    """
    return [random.choice(individuals) for i in range(k)]

    
def selTournament(individuals, k, tournsize, fit_attr="fitness"):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen

def selStochasticUniversalSamplingMinimize(individuals, k, fit_attr="fitness"):
    """Select the *k* individuals among the input *individuals*.
    The selection is made by using a single random value to sample all of the
    individuals by choosing them at evenly spaced intervals. The list returned
    contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :return: A list of selected individuals.

    This function uses the :func:`~random.uniform` function from the python base
    :mod:`random` module.
    """   
    
    Error=[]
    for i in range(len(individuals)):
        Error.append(float(individuals[i].fitness.getValues()[0]))
    
    Error=np.asarray(Error)
    Fitness = 1.0/Error
    sum_fits = np.sum(Fitness)
    
    SortIndex = np.argsort(Error)   
    Fitness_sorted = Fitness[SortIndex]
    s_inds = [individuals[i] for i in SortIndex]


    distance = sum_fits / float(k)
    start = random.uniform(0, distance)
    points = [start + i*distance for i in range(k)]

    chosen = []
    for p in points:
        i = 0
        sum_ = Fitness_sorted[i]
        while sum_ < p:
            i += 1
            sum_ += Fitness_sorted[i]
        chosen.append(s_inds[i])

    return chosen    
    
def selRouletteMinimize(individuals, k, fit_attr="fitness"):
    """Select *k* individuals from the input *individuals* using *k*
    spins of a roulette. The selection is made by looking only at the first
    objective of each individual. The list returned contains references to
    the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    .. warning::
       The roulette selection by definition cannot be used for minimization
       or when the fitness can be smaller or equal to 0.
    """
    Error=[]
    for i in range(len(individuals)):
        Error.append(float(individuals[i].fitness.getValues()[0]))
    
    Error=np.asarray(Error)
    Fitness = 1.0/Error
    sum_fits = np.sum(Fitness)
    
    SortIndex = np.argsort(Error)   
    Fitness_sorted = Fitness[SortIndex]
    s_inds = [individuals[i] for i in SortIndex]

    chosen = []
    for i in range(k):
        u = random.random() * sum_fits
        sum_ = 0
        index=0
        for ind in s_inds:
            sum_ += Fitness_sorted[index]
            if sum_ > u:
                chosen.append(ind)
                break
            index=index+1
    return chosen


###########################
## Cross-over methods
###########################           

def cxOnePoint(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)
    ind1[cxpoint:], ind2[cxpoint:] = ind2[cxpoint:], ind1[cxpoint:]

    return ind1, ind2


def cxTwoPoint(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2], ind1[cxpoint1:cxpoint2]

    return ind1, ind2

def cxUniform(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

def cxBlend(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        gamma = alpha * random.random()
        ind1[i] = (1. - gamma) * x1 + gamma * x2
        ind2[i] = gamma * x1 + (1. - gamma) * x2

    return ind1, ind2

def cxLocalBlend(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        error1 = ind1.LocalError['Combo'][i]
        error2 = ind2.LocalError['Combo'][i]
        ratio_error1 = error1/(error1+error2)
        ratio_error2 = error2/(error1+error2)
        
        #set first parameter
        RandN = random.random()
        if(RandN<ratio_error1):
           gamma = 1-(0.5/ratio_error1)*RandN
        else:
           gamma = 0.5 - (0.5/ratio_error2)*(RandN-ratio_error1)

        gamma = alpha * gamma

        ind1[i] = (1. - gamma) * x1 + gamma * x2

        #set second parameter
        RandN = random.random()
        if(RandN<ratio_error2):
           gamma = 1-(0.5/ratio_error2)*RandN
        else:
           gamma = 0.5 - (0.5/ratio_error1)*(RandN-ratio_error2)

        gamma = alpha * gamma

        ind2[i] = (1. - gamma) * x2 + gamma * x1

    return ind1, ind2

def cxLocalBlendExp(ind1, ind2, alpha):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        error1 = ind1.LocalError['Combo'][i]
        error2 = ind2.LocalError['Combo'][i]
        ratio_error1 = error1/(error1+error2)
        ratio_error2 = error2/(error1+error2)
        
        #set first parameter
        RandN = random.random()       
        coeffExp = -np.log(0.5)/ratio_error1
        gamma = np.exp(-coeffExp*RandN)
        gamma = alpha * gamma
        ind1[i] = float((1. - gamma) * x1 + gamma * x2)

        #set second parameter
        RandN = random.random()       
        coeffExp = -np.log(0.5)/ratio_error2
        gamma = np.exp(-coeffExp*RandN)
        gamma = alpha * gamma
        ind2[i] = float((1. - gamma) * x2 + gamma * x1)

    return ind1, ind2

def cxLocalErrorPropExchange(ind1, ind2):
    """Executes a blend crossover that modify in-place the input individuals.
    The blend crossover expects :term:`sequence` individuals of floating point
    numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param alpha: Extent of the interval in which the new values can be drawn
                  for each attribute on both side of the parents' attributes.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        error1 = ind1.LocalError['Combo'][i]
        error2 = ind2.LocalError['Combo'][i]
        ratio_error1 = error1/(error1+error2)
        ratio_error2 = error2/(error1+error2)
        
        #set first parameter
        RandN = random.random()
        if(RandN<ratio_error2):
            ind1[i] = x1
        else:
            ind1[i] = x2

        #set second parameter
        RandN = random.random()
        if(RandN<ratio_error2):
            ind2[i] = x1
        else:
            ind2[i] = x2

    return ind1, ind2

def cxOnePointLocal(ind1, ind2):
    """Executes a one point crossover on the input :term:`sequence` individuals.
    The two individuals are modified in place. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the
    python base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint = random.randint(1, size - 1)

    errorVec1 = ind1.LocalError['Combo']
    errorVec2 = ind2.LocalError['Combo']
    
    theperfectindividual = np.asarray(ind1.copy())
    ind2arr = np.asarray(ind2.copy())
    fiterV = errorVec2<errorVec1
    fiterV = fiterV.reshape(-1,)
    theperfectindividual[fiterV] = ind2arr[fiterV]
    
    ind1[cxpoint:]= theperfectindividual[cxpoint:]
    ind2[cxpoint:] = theperfectindividual[cxpoint:]

    return ind1, ind2


def cxTwoPointLocal(ind1, ind2):
    """Executes a two-point crossover on the input :term:`sequence`
    individuals. The two individuals are modified in place and both keep
    their original length.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.randint` function from the Python
    base :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)

    errorVec1 = ind1.LocalError['Combo']
    errorVec2 = ind2.LocalError['Combo']
    
    theperfectindividual = np.asarray(ind1.copy())
    ind2arr = np.asarray(ind2.copy())
    fiterV = errorVec2<errorVec1
    fiterV = fiterV.reshape(-1,)
    theperfectindividual[fiterV] = ind2arr[fiterV]
    
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1 = cxpoint2
        cxpoint2 = cxpoint1

    ind1[cxpoint1:cxpoint2] = theperfectindividual[cxpoint1:cxpoint2]
    ind2[cxpoint1:cxpoint2] =theperfectindividual[cxpoint1:cxpoint2]

    return ind1, ind2

def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    """Executes a simulated binary crossover that modify in-place the input
    individuals. The simulated binary crossover expects :term:`sequence`
    individuals of floating point numbers.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :param low: A value or a :term:`python:sequence` of values that is the lower
                bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that is the upper
               bound of the search space.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    .. note::
       This implementation is similar to the one implemented in the
       original NSGA-II C code presented by Deb.
    """
    size = min(len(ind1), len(ind2))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2

###########################
## Mutation methods
###########################            

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x
            if(individual[i]>xu):
                individual[i] = xu
            if(individual[i]<xl):
                individual[i] = xl
    return individual,    


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

def mutUniformFloat(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() < indpb:
            individual[i] = random.uniform(xl, xu)

    return individual,  


###########################
## Book-keeping and output
###########################  
def VizBestInd(population, folder, P, ModelParamTable, gen):
            
    P['BestIdxP']['AllAcceptedList'] = [1]*P['iterationNum']
    P['AllAcceptedList'] = [1]*P['iterationNum']
    
    Error= GI.get_combo_err(P, P['BestIdxGen'])
      
    figfilename = P['folder']+'VisualInspection/Viz_T'+str(P['HypP']['thread_num'])+'_G_' + str(gen) +'_Err_'+ '{:.0f}'.format(Error*1000)+'.png'
    hisfilename = P['folder']+'HistoryFileInspection/His_'+str(P['HypP']['thread_num'])+'_G_' + str(gen)+'_Err_'+'{:.0f}'.format(Error*1000)+'.his'
    GI.OutputImageAndHisFile(P['BestIdxP'], figfilename,hisfilename)

    if(P['verbose']):
        print('Combined Error for gen '+str(gen)+': '+str(Error))
                  
def outputParametersScatterPlot(Data, genNum, P):

    Parameter1 = 'Fault13_Slip'
    Parameter2 = 'Plug0_X'
 
    
    V1 = Data[Parameter1]
    V2 = Data[Parameter2]
    Error = Data['TotalError']
    
    
    generations = Data['generation']
    uniqueGens = np.unique(generations)
    
    plt.close('all')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    min1 = np.min(V1)
    min2 = np.min(V2)
    
    max1 = np.max(V1)
    max2 = np.max(V2)
    
    # make the grid
    p1 = np.linspace(min1,max1,100)
    p2 = np.linspace(min2,max2,100)
    p1,p2 = np.meshgrid(p1,p2)
    Errori = griddata((V1,V2),Error,(p1,p2),method='linear')
    
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    Item = ax.imshow(Errori, extent = [min1, max1, min2, max2], cmap='cool', origin='lower')
    plt.colorbar(Item, ax=ax)
    
    ax.scatter(V1, V2)

    ax.set_xlabel(Parameter1)
    ax.set_ylabel(Parameter2)
    ax.set_title('title')
    ax.set_xlim([min1, max1])
    ax.set_ylim([min2, max2])
    
    figfilename = P['folder']+'VisualInspection/VizParameters_T'+str(genNum)+'.png'
    
    fig.savefig(figfilename, dpi = 100,bbox_inches="tight")
    
    plt.close('all')

def Record2FileNSGA(folder, vNames, pop, P, ModelParamTable, gen, append=False):

    file = folder+'ParameterHistory.csv'
    smallfile = folder+'Mismatch.csv'

    colNames = ['generation']
    colNames.append('TotalError')
    colNames = colNames+['Grav_Error', 'Mag_Error', 'Tracer_Error', 'GT_Error', 'FaultIntersection_Error']
    colNames = colNames+list(vNames)
    
    LowestError = 999999
    LowestErrorParameters = 0
    for i in range(len(pop)):
        Parameters=np.asarray(pop[i])
        Err = np.asarray(pop[i].fitness.getValues())
        SumErr = np.mean(Err)
        genP = np.asarray(gen)
#        allItems=np.concatenate((genP.reshape(-1,1), Err.reshape(-1,1), Parameters.reshape(-1,1)))
        allItems=np.concatenate((genP.reshape(-1,1), SumErr.reshape(-1,1), Err.reshape(-1,1), Parameters.reshape(-1,1)))
        if(i==0):
            BigPandas = pd.DataFrame(columns=colNames)
        BigPandas.loc[i]=allItems.reshape((-1,))
        if(SumErr<LowestError):
            LowestError=SumErr
            LowestErrorParameters = np.asarray(pop[i])
    if(append==False):
        BigPandas.to_csv(file, index=False)
    else:
        BigPandas.to_csv(file, mode='a', header=False, index=False)
       
    MismatchData = BigPandas[['generation', 'TotalError', 'Grav_Error', 'Mag_Error', 'Tracer_Error', 'GT_Error', 'FaultIntersection_Error']].copy()
    for dt in P['DataTypes']:
        MismatchData[dt+'_Error'] = MismatchData[dt+'_Error']*P[dt+'_comboNormalizingFactor']

    if(append==False):
        MismatchData.to_csv(smallfile, index=False)
    else:
        MismatchData.to_csv(smallfile, mode='a', header=False, index=False)
        
    parametersViz(LowestErrorParameters, folder+'Viz_'+str(gen), P, ModelParamTable, gen)