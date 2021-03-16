# -*- coding: utf-8 -*-
"""
Created on September 16, 2020

@author: ahinoamp@gmail.com

This script includes methods for visualizing steps in the optimisation process,
as well as other post processing visuals.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
import GeneralInversionUtil as GI
import matplotlib

matplotlib.use('Agg')

def visualize_opt_step(filename, P):  
    '''Visualize a single optimisation step via a set of graphics'''    

    # set up the figure
    # determine the number of columns and rows
    if(P['HypP']['ErrorType']=='Global'):
        colsPerDataType = 5
    else:
        colsPerDataType = 6
           
    P['nDataTypes'] = len(P['DataTypes'])
    
    widthRatios = [5]*(colsPerDataType-2) + [6] + [6]
    figWidth = np.sum(widthRatios)
    
    fig, axs = plt.subplots(P['nDataTypes'], colsPerDataType, 
                            figsize=(figWidth, 5.*P['nDataTypes']), 
                            gridspec_kw={'width_ratios': widthRatios})
    
    plt.subplots_adjust(hspace = 0.15,)
    plt.subplots_adjust(wspace = 0.3,)

    # Determine the minimum and maximum extents of the plots
    if('Viz' not in list(P.keys())):
        P['Viz'] = {}
        P['Viz']['GeoX'] = np.linspace(P['xmin'], P['xmax'], 
                                       np.shape(P['Grav']['sim'])[1], dtype=float)

        P['Viz']['GeoY'] = np.linspace(P['ymin'], P['ymax'], 
                                       np.shape(P['Grav']['sim'])[0], dtype=float)  

        P['Viz']['GeoXX'], P['Viz']['GeoYY'] = np.meshgrid(P['Viz']['GeoX'], 
                                                           P['Viz']['GeoY'], 
                                                           indexing='xy')
    P['Viz']['FaultsXY'] = GI.GetFaultsXY(P)

    # Plot a row for each data type for the summary        
    if('Grav' in P['DataTypes']):
        PlotGravityRow(P, axs)

    if('GT' in P['DataTypes']):
        PlotGraniteTopRow(P, axs)

    if('Mag' in P['DataTypes']):
        PlotMagneticsRow(P, axs)
    
    if('Tracer' in P['DataTypes']):
        PlotTracerData(P, axs)

    if('FaultMarkers' in P['DataTypes']):
        PlotFaultMarkers(P, axs)
  
    PlotSummary(P, axs, colsPerDataType)          
 
    if(P['HypP']['jupyter']==True):
        from IPython.display import clear_output
        from IPython import display
        clear_output(wait=True)
        display.display(fig)
        
    fig.savefig(filename,dpi=100,bbox_inches='tight')
    plt.close(fig)

def PlotGravityRow(P, axs):
    '''Create a row of plots showing the observed, simulated, mismatch and
    optimisation progress of the gravity data'''
    
    grid_mismatch = griddata(np.array([P['Grav']['xObs'],P['Grav']['yObs']]).T, 
                             (P['Grav']['L1MismatchMatrix'][:, -1].reshape(-1,)), 
                             (P['Viz']['GeoXX'], P['Viz']['GeoYY']), method='linear')

    norm, norm_mis = get_norm(P, datatype = 'Grav')
       
    # 1. plot observed gravity
    #`````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[0]
    else:
        ax = axs[0,0]
    
    plt_scatter(P, P['Grav']['xObs'], P['Grav']['yObs'], P['Grav']['Obs'], ax,
            title='Observed gravity', norm=norm, edgecolors='face', s=30,
            yticks='on')

    # 2. plot simulated gravity
    #`````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[1]
    else:
        ax = axs[0, 1]
    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], P['Grav']['simViz'], 
                norm, ax, title='Simulated gravity')

    # 3. plot local weights if error type is local
    #`````````````````````````````````````````````

    if(P['HypP']['ErrorType']=='Local'):
        if(P['nDataTypes']==1):
            ax = axs[2]
        else:
            ax = axs[0,2]
        # choose the local weights for parameter 40
        LocalWeightChosen = P['Grav']['LocalErrorWeights'][:, 40]
        
        plt_scatter(P, P['Grav']['xObs'], P['Grav']['yObs'], LocalWeightChosen, ax,
            title='Weights for p40', 
            norm=matplotlib.colors.Normalize(vmin=np.min(LocalWeightChosen),
                                             vmax=np.max(LocalWeightChosen)))
       
        axisIdx=3
    else:
        axisIdx=2        

    # 4. plot mismatch
    #`````````````````

    if(P['nDataTypes']==1):
        ax = axs[axisIdx]
    else:
        ax = axs[0, axisIdx]
    
    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], grid_mismatch, 
                norm_mis, ax, title='Mismatch gravity')
              
    # 5. plot optimisation progress
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx+1]
    else:
        ax = axs[0, axisIdx+1]
    
    plt_optim_progress(P, ax, datatype='Grav')

def get_norm(P, datatype = 'Grav'):
    
    minObs = np.nanmin(P[datatype]['Obs'])
    maxObs = np.nanmax(P[datatype]['Obs'])

    minSim = np.nanmin(P[datatype]['simViz'])
    maxSim = np.nanmax(P[datatype]['simViz'])
  
    minMismatch = np.nanmin(P[datatype]['L1MismatchMatrix'][:, -1])
    maxMismatch = np.nanmax(P[datatype]['L1MismatchMatrix'][:, -1])

    minV = np.min([minObs, minSim])
    maxV = np.max([maxObs, maxSim])

    rangeData = maxV-minV
    rangeMismatch = maxMismatch - minMismatch
    if(rangeData>rangeMismatch):
        levels = np.linspace(minV, maxV, 8)
        diff = levels[1]-levels[0]
        levelsmis = np.arange(minMismatch, minMismatch+8*diff, diff)
    else:
        levelsmis = np.linspace(minMismatch, maxMismatch, 8)
        diff = levelsmis[1]-levelsmis[0]
        levels = np.arange(minV, minV+8*diff, diff)

    norm = matplotlib.colors.Normalize(vmin=minV,vmax=maxV)
    normmismatch = matplotlib.colors.Normalize(vmin=np.min(levelsmis),vmax=np.max(levelsmis))

    return norm, normmismatch

def plt_scatter(P, x, y, c, ax, title, norm, s=22, edgecolors='k', 
                fault_alpha = 1, fault_lw=1, fault_ballsize = 3, yticks='off'):

    cf = ax.scatter(x, y, c=c, cmap = 'jet', norm=norm, edgecolors=edgecolors,
                    s=s)
    plt.colorbar(cf, orientation = 'vertical', ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.ticklabel_format(style='sci', axis='x')
    AddFaults2Axis(P['Viz']['FaultsXY'], ax, alpha=fault_alpha, lw=fault_lw, 
                   ballsize=fault_ballsize)   
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])
    if(yticks == 'off'):
        ax.set_yticks([])

def plt_heatmap(P, xx, yy, zz, norm, ax, title):

    extent = [np.min(xx[:]), np.max(xx[:]), np.min(yy[:]), np.max(yy[:])]
    cf = ax.imshow(zz,extent=extent, cmap = 'jet',norm=norm, origin='lower')
    plt.colorbar(cf, orientation = 'vertical', ax=ax,fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.set_yticks([])
    ax.ticklabel_format(style='sci', axis='x')   
    AddFaults2Axis(P['Viz']['FaultsXY'], ax)
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])

def plt_optim_progress(P, ax, datatype):
    
    dictTitle = {'Grav': 'gravity', 'Mag': 'magnetics', 'GT': 'granite top',
                 'Tracer': 'Tracer', 'FaultMarkers': 'fault markers'}
    x=np.arange(1, len(P[datatype]['L1MismatchList'])+1)
    ax.plot(x, P[datatype]['L1MismatchList'])
    ax.set_title('Mismatch per iteration '+dictTitle[datatype])
    if(P['AllAcceptedList'][-1]==1):
        ax.scatter(len(P[datatype]['L1MismatchList']), P[datatype]['L1MismatchList'][len(P[datatype]['L1MismatchList'])-1], s=30, marker ='v', c='g')
    else:
        ax.scatter(len(P[datatype]['L1MismatchList']), P[datatype]['L1MismatchList'][len(P[datatype]['L1MismatchList'])-1], s=30, marker ='X', c='r')
    ax.set_ylabel('Mismatch')    
    ax.yaxis.tick_right()
    ax.set_xlabel('Iterations')    
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    asp /= (6./4.5)
    ax.set_aspect(asp)

def PlotGraniteTopRow(P, axs):
    '''Create a row of plots showing the observed, simulated, mismatch and
    optimisation progress of the granite top data'''
    
    norm, norm_mis = get_norm(P, datatype = 'GT')
       
    # 1. plot observed granite top
    #`````````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[1]
    else:
        ax = axs[1,0]
    
    plt_scatter(P, P['GT']['xObs'], P['GT']['yObs'], P['GT']['Obs'], ax,
            title='Observed Granite Top', norm=norm, s=38, edgecolors='k',
            fault_alpha = 0.35, fault_lw=1, fault_ballsize = 3)


    # 2. plot simulated granite top
    #``````````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[1]
    else:
        ax = axs[1, 1]

    plt_heatmap(P, P['xxLith'], P['yyLith'], P['GT']['simViz'], 
                norm, ax, title='Simulated Granite Top')
    # Add empty scatter points that show locations of granite top markers
    cf = ax.scatter(P['GT']['xObs'], P['GT']['yObs'], facecolors='none', 
                    edgecolors='k', s=30, cmap = 'jet')

    # 3. plot local weights if error type is local
    #`````````````````````````````````````````````
    if(P['HypP']['ErrorType']=='Local'):
        if(P['nDataTypes']==1):
            ax = axs[2]
        else:
            ax = axs[1,2]
        # choose the local weights for parameter 40
        LocalWeightChosen = P['GT']['LocalErrorWeights'][:, 88]
        
        plt_scatter(P, P['GT']['xObs'], P['GT']['yObs'], LocalWeightChosen, ax,
            title='Weights for p40', 
                        norm=matplotlib.colors.Normalize(vmin=np.min(LocalWeightChosen),
                                             vmax=np.max(LocalWeightChosen)))
       
        axisIdx=3
    else:
        axisIdx=2        

    # 4. plot mismatch
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx]
    else:
        ax = axs[1, axisIdx]
        
    grid_mismatch = griddata(np.array([P['GT']['xObs'],P['GT']['yObs']]).T, 
                         (P['GT']['L1MismatchMatrix'][:,-1].reshape(-1,)), 
                         (P['Viz']['GeoXX'], P['Viz']['GeoYY']), method='linear')

    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], grid_mismatch, 
                norm_mis, ax, title='Mismatch granite top')

    cf = ax.scatter(P['GT']['xObs'], P['GT']['yObs'], facecolors='none', 
                    edgecolors='k', s=30, cmap = 'jet')

    # 5. plot optimisation progress
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx+1]
    else:
        ax = axs[1, axisIdx+1]
    
    plt_optim_progress(P, ax, datatype='GT')


def PlotMagneticsRow(P, axs):
    '''Create a row of plots showing the observed, simulated, mismatch and
    optimisation progress of the magnetics data'''
    
    if('ObsGrid' not in P['Mag'].keys()):
        xy_obs = np.array([P['Mag']['xObs'],P['Mag']['yObs']]).T
        P['Mag']['ObsGrid'] = griddata(xy_obs, P['Mag']['Obs'], 
                                       (P['Viz']['GeoXX'],P['Viz']['GeoYY']), 
                                       method='linear')

    norm, norm_mis = get_norm(P, datatype = 'Mag')
       
    # 1. plot observed magnetics
    #`````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[0]
    else:
        ax = axs[2,0]
    
    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], P['Mag']['ObsGrid'], 
                norm, ax, title='Observed magnetics')

    # 2. plot simulated magnetics
    #`````````````````````````
    if(P['nDataTypes']==1):
        ax = axs[1]
    else:
        ax = axs[2, 1]
    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], P['Mag']['simViz'], 
                norm, ax, title='Simulated magnetics')

    # 3. plot local weights if error type is local
    #`````````````````````````````````````````````

    if(P['HypP']['ErrorType']=='Local'):
        if(P['nDataTypes']==1):
            ax = axs[2]
        else:
            ax = axs[2,2]
        # choose the local weights for parameter 30
        LocalWeightChosen = P['Mag']['LocalErrorWeights'][:, 30]
        
        plt_scatter(P, P['Mag']['xObs'], P['Mag']['yObs'], LocalWeightChosen, ax,
            title='Weights for p30',
                        norm=matplotlib.colors.Normalize(vmin=np.min(LocalWeightChosen),
                                             vmax=np.max(LocalWeightChosen)))
       
        axisIdx=3
    else:
        axisIdx=2        

    # 4. plot mismatch
    #`````````````````

    if(P['nDataTypes']==1):
        ax = axs[axisIdx]
    else:
        ax = axs[2, axisIdx]
 
    grid_mismatch = griddata(np.array([P['Mag']['xObs'],P['Mag']['yObs']]).T, 
                         (P['Mag']['L1MismatchMatrix'][:,-1].reshape(-1,)), 
                         (P['Viz']['GeoXX'], P['Viz']['GeoYY']), method='linear')

    plt_heatmap(P, P['Viz']['GeoXX'], P['Viz']['GeoYY'], grid_mismatch, 
                norm_mis, ax, title='Mismatch Magnetics')
              
    # 5. plot optimisation progress
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx+1]
    else:
        ax = axs[2, axisIdx+1]
    
    plt_optim_progress(P, ax, datatype='Mag')

    norm, norm_mis = get_norm(P, datatype = 'Mag')

def PlotTracerData(P, axs):
    '''Create a row of plots showing the observed, simulated, mismatch and
    optimisation progress of the tracer data'''
    
    # Plot observed data
    ax = axs[3,0]
    AddFaults2Axis(P['Viz']['FaultsXY'], ax,  alpha=0.355, lw=1, ballsize=3)
    Xarrow = P['Tracer']['Connections']['midx'].values
    Yarrow = P['Tracer']['Connections']['midy'].values
    X1 = P['Tracer']['Connections']['x1']
    Y1 = P['Tracer']['Connections']['y1']
    X2 = P['Tracer']['Connections']['x2']
    Y2 = P['Tracer']['Connections']['y2']
    U = P['Tracer']['Connections']['dx'].values
    V = P['Tracer']['Connections']['dy'].values
    ax.scatter(X1, Y1, c='red', s=5)
    ax.scatter(X2, Y2, c='blue', s=5)
    cf = ax.quiver(X1, Y1, U, V, scale=1,units='xy')
    ax.set_title('Observed tracer')
    ax.set_aspect('equal', 'box')
    ax.ticklabel_format(style='sci', axis='x')
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])

    # Plot simulated data
    ax = axs[3,1]
    AddFaults2Axis(P['Viz']['FaultsXY'], ax,  alpha=0.35, lw=1, ballsize=3)
    ax.scatter(X1, Y1, c='red', s=5, label='Injection')
    ax.scatter(X2, Y2, c='blue', s=5, label='Collection')
    filterC = (P['TracersConnected']>0).reshape(-1,)
    C = P['TracersConnected']
    ax.quiver(X1[filterC], Y1[filterC], U[filterC], V[filterC],color='lime', scale=1,units='xy', label='Connected')
    ax.quiver(X1[~filterC], Y1[~filterC], U[~filterC], V[~filterC],color='dimgray', scale=1,units='xy', label='Not connected')
    ax.legend(loc='upper left')
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])
    ax.set_title('Simulated tracer')
    ax.set_aspect('equal', 'box')
    ax.set_yticks([])
    ax.ticklabel_format(style='sci', axis='x')   

    # Plot extra data
    if(P['HypP']['ErrorType']=='Local'):
        ax = axs[3, 2]
        ax.set_xlim([P['xmin'], P['xmax']])
        ax.set_ylim([P['ymin'], P['ymax']])
        axisIdx=3
    else:
        axisIdx=2        

    ax = axs[3,axisIdx]
    index = np.arange(1, len(P['TracersConnected'])+1) 
    values=np.ones((len(P['TracersConnected']),))
    colors = [(0.4, 1, 0), (0.8, 0.8, 0.8)]
    colorV = np.asarray([colors[0]]*len(P['TracersConnected']))
    filterNotC = (P['TracersConnected']<1).reshape(-1,)
    colorV[filterNotC]=colors[1]
    ax.bar(index, values, color=colorV)
    ax.set_title('Mismatch tracer')
    ax.set_yticks([])
    ax.ticklabel_format(style='sci', axis='x')
    listNames = ['Connected', 'Not connected']    
    custom_lines = [Line2D([0], [0], color=colors[i], lw=8) for i in range(len(listNames))]   
    ax.legend(custom_lines, listNames, loc='upper left')       
        
    # 5. plot optimisation progress
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx+1]
    else:
        ax = axs[3, axisIdx+1]
    
    plt_optim_progress(P, ax, datatype='Tracer')


def PlotFaultMarkers(P, axs):
    '''Create a row of plots showing the observed, simulated, mismatch and
    optimisation progress of the fault marker data'''

    # 1. plot observed data
    orient='vertical'
    ax = axs[4, 0]
    #first plot the wells
    x = P['FaultMarkers']['WellData']['Xm'].values
    y = P['FaultMarkers']['WellData']['Ym'].values
    z = P['FaultMarkers']['WellData']['Zm']   
    idwell = P['FaultMarkers']['WellData']['id']   
    AddFaults2Axis(P['Viz']['FaultsXY'], ax,  alpha=0.355, lw=1, ballsize=3)
    cf = ax.scatter(x, y, c = 'k', s=10)
    xmarker = P['FaultMarkers']['Obs']['X'].values
    ymarker = P['FaultMarkers']['Obs']['Y'].values
    zmarker = P['FaultMarkers']['Obs']['Z'].values
    idmarker = P['FaultMarkers']['Obs']['wellid'].values
    cf = ax.scatter(xmarker, ymarker, c = 'r', s=10, marker='X')
    ax.set_title('Observed Fault Intersections')
    ax.set_aspect('equal', 'box')
#    ax.set_yticks([])
    ax.ticklabel_format(style='sci', axis='x')
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])

    # 2. Plot the simulated data    
    ax = axs[4, 1]
    #first plot the wells
    AddFaults2Axis(P['Viz']['FaultsXY'], ax,  alpha=0.355, lw=1, ballsize=3)
    cf = ax.scatter(x, y, c='k', s=8)   
    cf = ax.scatter(P['FaultMarkers']['simX'], P['FaultMarkers']['simY'], c = 'r', s=8, marker='X')
    xWell = P['topXWell']
    yWell = P['topYWell']
    idWell = P['idPlotWell']
    for i in range(len(yWell)):
        ax.text(xWell[i], yWell[i], str(idWell[i]), color='b', fontsize=10)   
    ax.set_title('Observed Fault Intersections')
    ax.set_aspect('equal', 'box')
    ax.set_yticks([])
    ax.ticklabel_format(style='sci', axis='x')
    ax.set_xlim([P['xmin'], P['xmax']])
    ax.set_ylim([P['ymin'], P['ymax']])


    #3. Plot some extra stuff
    if(P['HypP']['ErrorType']=='Local'):
        ax = axs[4, 2]
        axisIdx=3
    else:
        axisIdx=2 

    ax = axs[4,axisIdx]
    ax.plot(P['idWells'], P['zWells'])
    ax.scatter(P['FaultMarkers']['simID'], P['FaultMarkers']['simZ'], s=8, c='r', marker='X', alpha=0.7)
    ax.scatter(idmarker, zmarker, c = 'k', s=10)
    ax.set_title('Mismatch Fault Intersections')

    # 5. plot optimisation progress
    #`````````````````
    if(P['nDataTypes']==1):
        ax = axs[axisIdx+1]
    else:
        ax = axs[4, axisIdx+1]
    
    plt_optim_progress(P, ax, datatype='FaultMarkers')

def PlotSummary(P, axs, rowsPerDataType):

    if(P['nDataTypes']==1):
        ax = axs[rowsPerDataType-1]
    else:
        ax = axs[0, rowsPerDataType-1]

    comboMismatch = GI.get_combo_err_list(P)
    x=np.arange(1, len(comboMismatch)+1)

    # Plot the error of each data type
    for dt in P['DataTypes']:
        if(P['HypP']['ErrorType']=='Global'):
            ax.plot(x, np.asarray(P[dt]['L1MismatchList'])/P['DatNormCoef'][dt], alpha=0.4)
        else:
            ax.plot(x, np.asarray(P[dt]['L1MismatchList'])/np.mean(P['DatNormCoef'][dt]), alpha=0.4)
    
    # Plot the combined error per iteration
    ax.plot(x, comboMismatch, color='r', lw=3)
    ax.set_title('Mismatch per iteration combo')

    #Add dots based on acceptance
    if(P['AllAcceptedList'][-1]>=0.5):
        ax.scatter(len(comboMismatch), comboMismatch[len(comboMismatch)-1],
                   s=50, marker ='v', c='g')
    else:
        ax.scatter(len(comboMismatch), comboMismatch[len(comboMismatch)-1], 
                   s=50, marker ='X', c='r')
            
    ax.set_ylabel('Mismatch')    
    ax.yaxis.tick_right()
    ax.set_xlabel('Iterations')    
    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    asp /= (6./4.5)
    ax.set_aspect(asp)

    if(P['nDataTypes']==1):
        return

    # Create parallel coordinate plots
    ax = axs[1,rowsPerDataType-1]
    
    dictPd = {}
    for dt in P['DataTypes']:
        dictPd[dt] = np.asarray(P[dt+'']['L1MismatchList'])/P['DatNormCoef'][dt] 
    ax.yaxis.tick_right()

    df = pd.DataFrame(dictPd)
    df['ErrorPerType'] = 'Previous'
    df.loc[len(df)-1, 'ErrorPerType'] = 'Current'
    
    if(len(df)>30):
        alpha = 0.1
    else:
        alpha = 0.2

    if(len(df)==1):
        pd.plotting.parallel_coordinates( df, 'ErrorPerType', 
                                     color=[[0.9, 0.2, 0.2, 1.0]],
                                     linewidth=2, ax=ax)        
    else:
        pd.plotting.parallel_coordinates( df, 'ErrorPerType', 
                                     color=[[0.1, 0.1, 0.5, alpha],[0.9, 0.2, 0.2, 1.0]],
                                     linewidth=2, ax=ax)

    asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
    asp /= (6./4.5)
    ax.set_aspect(asp)
    
    if((P['HypP']['OptimMethod']!='MCMC')|(P['iterationNum']<(P['HypP']['nExploreRuns']+1))):
        return
        
    #Plot the acceptance rate 
    ax = axs[2,rowsPerDataType-1]
    window=np.min([20, len(P['AllAcceptedList'])])
    cumAcceptanceRate = pd.Series(P['AllAcceptedList']).rolling(window=5, min_periods=1).mean()

    x = np.arange(1, (len(cumAcceptanceRate)+1))
    ax.plot(x, cumAcceptanceRate)
    ax.set_ylabel('Acceptance rate (last 20)')    
    ax.set_xlabel('Iterations')    
    ax.yaxis.tick_right()
    
    #Take note of a few extra stuff 
    ax = axs[3,rowsPerDataType-1]
    ax.yaxis.tick_right()
    listTxt = []

    curr_err = comboMismatch[-1]
    listTxt.append('Current error: '+'{:.3f}'.format(curr_err))

    if(P['HypP']['ErrorType']=='Global'):
        lastAcceptedIndex = P['lastAcceptedIdx']   
        last_accepted_err = comboMismatch[lastAcceptedIndex]
        s='Last accepted error: '+'{:.3f}'.format(last_accepted_err) +' at index '+str(lastAcceptedIndex)
        listTxt.append(s)
    elif(P['HypP']['ErrorType']=='Local'):
        currentAcceptReject = P['AllAcceptedMatrix'][:, -1]
        numberAccepted = int(np.sum(currentAcceptReject))
        totalParameters = P['nParam']
        percent = P['AllAcceptedList'][-1]
        s = 'Accepted: ' + str(numberAccepted) +'/' + str(totalParameters) + '=' + '{:.2f}'.format(percent) 
        listTxt.append(s)
    
    if(P['HypP']['AcceptProbType']!="Annealing"):
        temperature = np.mean(P['lastNormFactor'])
        listTxt.append('Temperature: '+'{:.3f}'.format(temperature))
        
        
    if((P['HypP']['AcceptProbType']!="Error must decrease") & (P['HypP']['ErrorType']=='Global')):
        acceptance_prob = P['lastAcceptanceProbability']
        listTxt.append('Acceptance Prob: '+'{:.3f}'.format(acceptance_prob))
        temperature = P['lastNormFactor']
        listTxt.append('Temperature: '+'{:.3f}'.format(temperature))

    
    positions = np.linspace(len(listTxt)*3+3, 0, len(listTxt)+2)
    
    ax.set_xlim([0, 20])
    ax.set_ylim([np.min(positions), np.max(positions)])
    ax.yaxis.tick_right()

    for i in range(len(listTxt)):
        ax.text(1,positions[i+1],listTxt[i])
    
    
def AddFaults2Axis(FaultsXY, ax, alpha=1, lw=2, ballsize=8):
    '''Draw the faults on an axis together with fault balls'''
    nFaults = len(FaultsXY['x'])
    for i in range(nFaults):
        x=FaultsXY['x'][i]
        y=FaultsXY['y'][i]
        ax.plot(x,y, color='k', alpha=alpha, linewidth=lw)
        ax.scatter(FaultsXY['fault_ballsX'][i], FaultsXY['fault_ballsY'][i], color='k', s=ballsize, alpha=alpha)


#Make uniparameter plots
def GenerateUniparameterPlots(folder):
    ModelParamTableF = folder+'ThreadModelParameters.csv'
    PriorUncertaintyTable = pd.read_csv(ModelParamTableF)
    
    ParameterTable = pd.read_csv(folder+'ParameterHistory.csv')
    ParameterTable = ParameterTable.drop(ParameterTable.columns[0], axis=1)
    ParameterTable = ParameterTable.drop(ParameterTable.columns[0], axis=1)
    AllAcceptedList = pd.read_csv(folder+'AllAcceptedList.csv')
    AllAcceptedList = AllAcceptedList['AllAcceptedList'].values
    nParam=np.shape(ParameterTable)[1]
    
    RealizationNumber=np.arange(len(ParameterTable))
    for i in range(nParam):
        fig, axs = plt.subplots(1, 1, figsize=(6,3))
        ParameterValue = ParameterTable.iloc[:, i]
        parameterName = ParameterTable.columns.tolist()[i]
        event = int(parameterName[(parameterName.find("Event") + len("Event")):parameterName.find("Prop")])
        prop =  parameterName[(parameterName.find("Prop") + len("Prop")):parameterName.find("SubEventName")]
        eventname = parameterName[(parameterName.find("SubEventName") + len("SubEventName")):]
        scenario = int(parameterName[(parameterName.find("Scenario") + len("Scenario")):parameterName.find("Event")])
        filterV = ((PriorUncertaintyTable['EventNumber']==event)&
                   (PriorUncertaintyTable['Prop']==prop)&
                   (PriorUncertaintyTable['EventName']==eventname)&
                   (PriorUncertaintyTable['Scenario']==scenario))
        axs.scatter(ParameterValue, RealizationNumber)
        p_min = PriorUncertaintyTable.loc[filterV, 'minV'].values[0]
        p_max = PriorUncertaintyTable.loc[filterV, 'maxV'].values[0]
        axs.set_xlim([p_min, p_max])
        axs.set_xlabel(parameterName)
        axs.set_ylabel('Realization Number')
        axs.set_title('Progression of '+parameterName)
        fig.savefig('ScratchPlots/UniParameter/'+parameterName+'.png',dpi=100,bbox_inches='tight')
        plt.close(fig)

#plot the path of some parameters

def Generate2D_Map(folder, pNum1 = 14, pNum2 = 15):
   
    MismatchListPD = pd.read_csv(folder+'Mismatch.csv')
    ParameterTable = pd.read_csv(folder+'ParameterHistory.csv')
    ParameterTable = ParameterTable.drop(ParameterTable.columns[0], axis=1)
    ParameterTable = ParameterTable.drop(ParameterTable.columns[0], axis=1)
    AllAcceptedList = pd.read_csv(folder+'AllAcceptedList.csv')
    AllAcceptedList = AllAcceptedList['AllAcceptedList'].values

    n_threads = 1
    AllNumberRunsVals = [len(MismatchListPD)]
    AllThreadNumber = [0]*len(MismatchListPD)
    VarA_Vals = ParameterTable.iloc[:, pNum1]
    VarB_Vals = ParameterTable.iloc[:, pNum2]
    VarA_name = ParameterTable.columns.tolist()[pNum1]
    VarB_name = ParameterTable.columns.tolist()[pNum2]
    VarA_min = np.min(VarA_Vals)
    VarB_min = np.min(VarB_Vals)
    VarA_max = np.max(VarA_Vals)
    VarB_max = np.max(VarB_Vals)
    alphaVal = 0.3
    MakeSearchMap(n_threads, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, VarA_Vals, VarB_Vals, VarA_min, VarB_min,\
                      VarA_max, VarB_max, VarA_name, VarB_name, alphaVal, folder)


def GenerateVisualizations(P):
    errorStdList =[]
    error = GI.get_combo_err_list(P)
    for i in range(len(error)):
        indexS = np.min([len(error), 20])
        normalizingFactor = np.std(error[-indexS:])
        errorStdList.append(normalizingFactor)

    MismatchListPD = pd.DataFrame({'Mismatch': error})
        
    PlotErrorProgression(MismatchListPD, P['AllAcceptedList'], errorStdList, P['folder'])
    #Generate2D_Map(pNum1 = 14, pNum2 = 15, folder=P['folder'])
    #Viz.GenerateUniparameterPlots()
    ''
    
def MakeSearchMap(n_threads, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, VarA_Vals, VarB_Vals, VarA_min, VarB_min,\
                  VarA_max, VarB_max, VarA_name, VarB_name, alphaVal, folder, ax=0):
    
    StartI_List = [0]
    EndI_List = []
    Lines = []
    ScatterMarker = []
    LineColors = []
    ScatterX = []
    ScatterY = []
    
    startI = 0    
#    print('The number of stuff is ' + str(len(AllAcceptedList)))
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
     
    ax.scatter(ScatterX[ScatterMarker == 0], ScatterY[ScatterMarker == 0], marker='>', c='g', zorder=8)
    ax.scatter(ScatterX[ScatterMarker == 1], ScatterY[ScatterMarker == 1], marker='o', c='b', zorder=7, alpha = 0.5)
    ax.scatter(ScatterX[ScatterMarker == 2], ScatterY[ScatterMarker == 2], marker='x', c='r',zorder=6, alpha = 0.5)
    ax.scatter(ScatterX[ScatterMarker == 3], ScatterY[ScatterMarker == 3], marker='s', c='k', zorder=9)
    
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
   
    f.savefig(folder+'FinalSearchMap_'+VarA_name+'_and_'+VarB_name+'.png',dpi=300,bbox_inches='tight')
    plt.close(f)
    
def PlotErrorProgression(MismatchListPD, AllAcceptedList, errorStdList, folder):    
    
    fig, axs = plt.subplots(1, 3, figsize=(10,4))
    ax = axs[0]
    ax.plot(MismatchListPD['Mismatch'])
    ax.set_title('mismatch (mGal)')
    
    ax = axs[1]
    cumAcceptanceRate = np.cumsum(AllAcceptedList)/ np.arange(1,len(AllAcceptedList)+1)
    ax.plot(cumAcceptanceRate)
    ax.set_title('Acceptance')
    
    ax = axs[2]
    ax.plot(errorStdList)
    ax.set_title('errorStdList')
    
    fig.savefig(folder+'MismatchAcceptRate.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    