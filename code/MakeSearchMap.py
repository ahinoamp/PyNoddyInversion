# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:00:49 2019

@author: ahinoamp
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

def MakeSearchMap(n_threads, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, VarA_Vals, VarB_Vals, VarA_min, VarB_min,\
                  VarA_max, VarB_max, VarA_name, VarB_name, alphaVal):
    
    StartI_List = [0]
    EndI_List = []
    Lines = []
    ScatterMarker = []
    LineColors = []
    ScatterX = []
    ScatterY = []
    
    startI = 0    
    print('The number of stuff is ' + str(len(AllAcceptedList)))
    for sim_index in range(n_threads):
        print('Ive reached thread index ' + str(sim_index))
        LastIndexAccepted = startI
        NumberRuns = int(AllNumberRunsVals[sim_index])
              
        for iteration in range (startI, startI+NumberRuns):
        
            ScatterX.append(VarA_Vals[iteration])
            ScatterY.append(VarB_Vals[iteration])
            
            if(AllAcceptedList[iteration]>0.5):
                pt1 = (VarA_Vals[LastIndexAccepted], VarB_Vals[LastIndexAccepted])
                pt2 = (VarA_Vals[iteration], VarB_Vals[iteration])               
                Lines.append((pt1,pt2))
                ScatterMarker.append(1) # 0 is start, 1 is accepted, 2 is rejected and 3 is end
                LastIndexAccepted = iteration
                LineColors.append('b')
            else:
                pt1 = (VarA_Vals[LastIndexAccepted], VarB_Vals[LastIndexAccepted])
                pt2 = (VarA_Vals[iteration], VarB_Vals[iteration])  
                Lines.append((pt1,pt2))
                LineColors.append('r')
                ScatterMarker.append(2)
        
        startI = startI + NumberRuns
        StartI_List.append(startI)
        EndI_List.append(startI-1)   
    
    for sim_index in range(n_threads):            
        startItmp = StartI_List[sim_index]
        endItmp = EndI_List[sim_index]
        ScatterMarker[startItmp] = 0
        ScatterMarker[endItmp] = 3

    f, ax = plt.subplots(1)                  
    colored_lines = LineCollection(Lines, colors=LineColors, linewidths=(1,), alpha = 0.5)
    ax.add_collection(colored_lines)
    ax.autoscale_view()
    
    ScatterX = np.asarray(ScatterX)
    ScatterY = np.asarray(ScatterY)
    ScatterMarker = np.asarray(ScatterMarker)   
     
    plt.scatter(ScatterX[ScatterMarker == 0], ScatterY[ScatterMarker == 0], marker='>', c='g', zorder=8)
    plt.scatter(ScatterX[ScatterMarker == 1], ScatterY[ScatterMarker == 1], marker='o', c='b', zorder=7, alpha = 0.5)
    plt.scatter(ScatterX[ScatterMarker == 2], ScatterY[ScatterMarker == 2], marker='x', c='r',zorder=6, alpha = 0.5)
    plt.scatter(ScatterX[ScatterMarker == 3], ScatterY[ScatterMarker == 3], marker='s', c='k', zorder=9)
    
    ax.set_ylim(ymin=VarB_min)
    ax.set_ylim(ymax=VarB_max)
    ax.set_xlim(xmin=VarA_min)
    ax.set_xlim(xmax=VarA_max)
    ax.set_xlabel(VarA_name)
    ax.set_ylabel(VarB_name)
    ax.set_title('Searching the '+VarA_name+' and '+VarB_name+' space')
    MarkerSize = 8
    legend_elements = [Line2D([0], [0], marker='>', color='g', markerfacecolor='g', markersize=MarkerSize, label='Start'),
               Line2D([0], [0], marker='o', color='b', markerfacecolor='b', markersize=MarkerSize, label='Accepted'),
               Line2D([0], [0], marker='x', color='r', markerfacecolor='r', markersize=MarkerSize, label='Rejected'),
               Line2D([0], [0], marker='s', color='k', markerfacecolor='k', markersize=MarkerSize, label='End')]
    ax.legend(handles=legend_elements, loc='upper right').set_zorder(102)

    f.savefig('FinalSearchMap_'+VarA_name+'_and_'+VarB_name+'.png',dpi=300,bbox_inches='tight')
    plt.close(f)