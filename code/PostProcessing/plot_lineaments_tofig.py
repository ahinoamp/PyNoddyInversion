# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:52:30 2020

@author: ahinoamp
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.close('all')

P={}
xy_origin=[316448, 4379166, -2700]
xy_extent = [8850, 9035,3900]
P['xy_origin']=xy_origin
P['xy_extent'] = xy_extent

xsection = P['xy_extent'][0]/2.0 
ysection = P['xy_extent'][1]/2.0 
zsection = 0 

fz = 18

directions = ['X', 'Y', 'Z']

OneLargeDict = pd.read_pickle('PriorPosteriorFaultLineaments.pkl')
xPriorL = OneLargeDict['xLargePri']
zPriorL = OneLargeDict['zLargePri']

OneLargeDict = pd.read_pickle('PosteriorFaultLineaments.pkl')
xPostL = OneLargeDict['xLargePost']
zPostL = OneLargeDict['zLargePost']

for d in range(len(directions)):
    print(str(d))
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    direction = directions[d]

    xPri = xPriorL[direction]
    zPri = zPriorL[direction]

    xPost = xPostL[direction]
    zPost = zPostL[direction]

    ax.plot(xPost, zPost, color=[0.1, 0.1, 0.9], alpha=0.2,zorder=10000)
    ax.plot(xPri, zPri, color=[0.9, 0.3, 0.3], alpha=0.1, zorder=-1)
    if(direction=='X'):
        fz=12
        ax.set_xlabel('Y (m)', fontsize=fz)
        ax.set_ylabel('Z (m)', fontsize=fz)
        ax.set_title('Prior vs. posterior fault models (x = '+str(np.round(xsection))+') cross section', fontsize=fz)
    if(direction=='Y'):
        fz=12
        ax.set_xlabel('X (m)', fontsize=fz)
        ax.set_ylabel('Z (m)', fontsize=fz)
        ax.set_title('Prior vs. posterior fault models (y = '+str(np.round(ysection))+') cross section', fontsize=fz)
    if(direction=='Z'):
        fz=17
        ax.set_xlabel('X (m)', fontsize=fz)
        ax.set_ylabel('Y (m)', fontsize=fz)
        ax.set_title('Prior vs. posterior fault models (z = '+str(np.round(zsection))+') cross section', fontsize=fz)
        ax.plot([xsection+P['xy_origin'][0], xsection+P['xy_origin'][0]], [P['xy_origin'][1], P['xy_origin'][1]+P['xy_extent'][1]], color='k', ls=':')
        ax.plot([P['xy_origin'][0], P['xy_origin'][0]+P['xy_extent'][0]], [ysection+P['xy_origin'][1], ysection+P['xy_origin'][1]], color='k', ls='--')
    
    ax.set_aspect('equal')
    if(direction=='X'):
        ax.set_xlim([P['xy_origin'][1], P['xy_origin'][1]+P['xy_extent'][1]])
        ax.set_ylim([P['xy_origin'][2], P['xy_origin'][2]+P['xy_extent'][2]])
    if(direction=='Y'):
        ax.set_xlim([P['xy_origin'][0], P['xy_origin'][0]+P['xy_extent'][0]])
        ax.set_ylim([P['xy_origin'][2], P['xy_origin'][2]+P['xy_extent'][2]])
    if(direction=='Z'):
        ax.set_xlim([P['xy_origin'][0], P['xy_origin'][0]+P['xy_extent'][0]])
        ax.set_ylim([P['xy_origin'][1], P['xy_origin'][1]+P['xy_extent'][1]])
    
    if(direction=='Z'):
        listLines = ['Prior', 'Posterior', 'X cross section', 'Y cross section']    
        colors1 = [[0.9, 0.3, 0.3, 0.8], [0.1, 0.1, 0.9, 0.8]]
        colors2= [[0, 0, 0, 1],[0, 0, 0, 1]]
        linestyles = [':', '--']
        custom_lines = [Line2D([0], [0], color=colors1[i], lw=4) for i in range(2)]   
        custom_lines = custom_lines+[Line2D([0], [0], color=colors2[i], lw=2, ls=linestyles[i]) for i in range(2)]   
        ax.legend(custom_lines, listLines, bbox_to_anchor=(1.02,1.02), loc="upper left", fontsize=11)    
    else:
        listLines = ['Prior', 'Posterior']    
        colors = [[0.9, 0.3, 0.3, 0.8], [0.1, 0.1, 0.9, 0.8]]
        custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(listLines))]   
        ax.legend(custom_lines, listLines, bbox_to_anchor=(1.02,1.02), loc="upper left", fontsize=11)

    if(direction!='X'):
        ax.set_xticks([317000, 319000, 321000, 323000, 325000])
    
    fig.savefig('hope'+direction+'.png', dpi = 300,bbox_inches="tight")