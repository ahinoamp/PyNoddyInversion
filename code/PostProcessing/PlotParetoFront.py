# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:51:14 2020

@author: ahinoamp

Plot the progression of the grav/tracer error pareto front for NSGA results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load the data
Data = pd.read_csv('ParameterHistory.csv')

uniqueGens = np.unique(Data['generation'])

for i in range(len(uniqueGens)):
    filterG = Data['generation']==uniqueGens[i]
    Datai = Data[filterG]

    Grav_Errori = Datai['Grav_Error']
    Tracer_Errori = Datai['Tracer_Error']
    
    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
    ax.scatter(Grav_Errori, Tracer_Errori)

    ax.set_xlabel('Gravity Error')
    ax.set_ylabel('Tracer Error')

    ax.set_xlim([0.5,1.1])
    ax.set_ylim([0,1.2])
    
    ax.set_title('NSGA fronts')    
    fig.savefig('PlotFrontier/Gen'+str(i)+'.png', dpi = 100,bbox_inches="tight")

    plt.close('all')    
    