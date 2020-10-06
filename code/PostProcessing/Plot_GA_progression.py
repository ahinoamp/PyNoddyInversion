# -*- coding: utf-8 -*-
"""
Created on Wed May 27 08:51:06 2020

@author: ahinoamp

Plot parameter and error progression for Genetic Algorithm

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import glob
import re
import GravityInversionUtilities as GI
import LoadInputDataUtility as DI
import VisualizationUtilities as Viz
from mpl_toolkits.axes_grid1 import make_axes_locatable
file = 'ParameterHistory.csv'

Data = pd.read_csv(file)

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

#Calculate the minimumPerGeneration
MinErrorPerGen = Data[Data['TotalError'] == Data.groupby('generation')['TotalError'].transform('min')]
MinErrorPerGen = MinErrorPerGen['TotalError'].values

# make the grid
p1 = np.linspace(min1,max1,100)
p2 = np.linspace(min2,max2,100)
p1,p2 = np.meshgrid(p1,p2)
Errori = griddata((V1,V2),Error,(p1,p2),method='linear')



for i in range(len(uniqueGens)):
    fig, axs = plt.subplots(1, 2, figsize=(16,8))
    ax=axs[0]
    im = ax.imshow(Errori, extent = [min1, max1, min2, max2], cmap='jet', 
                     origin='lower', alpha=0.3)
    plt.colorbar(im,fraction=0.046, pad=0.04, ax=ax)
    g = uniqueGens[i]
    filterV = generations==g
    
    BestModel = np.argmin(Error[filterV])
    if(i>0):
        prevGen = g-1
        filterPrev = generations==prevGen
        ax.scatter(V1[filterPrev], V2[filterPrev],color='tab:pink', marker='x', s=30, label='Previous generation')
    ax.scatter(V1[filterV], V2[filterV],color='r', s=25,label='Current generation')
    ax.scatter(V1[filterV][BestModel], V2[filterV][BestModel], 
               s=200, edgecolors='r', linewidths =1, color='k', marker='*', label='Best Model')


    ax.set_xlabel(Parameter1)
    ax.set_ylabel(Parameter2)
    
    ax.set_title('Genetic algorithm with uniform exchange mating')
    ax.set_xlim([min1, max1])
    ax.set_ylim([min2, max2])
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    ax.set_aspect(asp)
    ax.legend(loc ='upper right', prop={'size': 10})

    ax = axs[1]
    MinErrorPerGeni = MinErrorPerGen[0:i+1]
    if(i==0):
        ax.scatter([0], MinErrorPerGeni, color='r')
    else:
        x=np.arange(1, len(MinErrorPerGeni)+1)
        ax.plot(x, MinErrorPerGeni, color='r', lw=3)
    ax.set_title('Mismatch per iteration')
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    asp /= (6./4.5)
    ax.set_aspect(asp)   
    fig.savefig('PlotGA/'+str(i)+'.png', dpi = 100,bbox_inches="tight")

    plt.close('all')


# target grid to interpolate to


# set mask
