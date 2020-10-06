# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:10:19 2020

@author: ahinoamp
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import VisualizationUtilities as Viz

folder= 'Combo_Scratch/Thread0/'
#folder= 'Z:/MCMC_Scratch/Thread52/'
#ErrorPerParameter_History = np.loadtxt(folder+'ErrorPerParameter_History', delimiter=',') 

GlobalError = pd.read_csv(folder+'Mismatch.csv')

AllAcceptedMatrix = np.loadtxt(folder+'AllAcceptedMatrix', delimiter=',') 

ParameterHistory = pd.read_csv(folder+'ParameterHistory.csv')
ParameterNames = ParameterHistory['EventName']+'_'+ParameterHistory['Property']

nParameters = np.shape(ParameterHistory)[0]
idxStartRealizations = ParameterHistory.columns.tolist().index('0')
nIterations = np.shape(ParameterHistory)[1]-idxStartRealizations



ErrorPerParameter_History_average = np.mean(ErrorPerParameter_History, axis=1)
fig, ax = plt.subplots(1, 1, figsize=(8,8))

ax.plot(ErrorPerParameter_History_average)

fig.savefig(folder+'MeanLocals.png', dpi = 60,bbox_inches="tight")
plt.close('all')


ploton = 0
if(ploton==1):
    for i in range(nParameters):
        
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
        ax.plot(ErrorPerParameter_History[:, i])
        
        ax.set_title(ParameterNames[i])
        ax.set_xlabel(ParameterNames[i])
        ax.set_ylabel('mGal (error)')
        fig.savefig(folder+'Parameter'+str(i)+'.png', dpi = 60,bbox_inches="tight")
        plt.close('all')

ploton = 0
ErrValues = GlobalError['Mismatch']
if(ploton==1):
    for i in range(nParameters):
        
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
        ParaValues=ParameterHistory.iloc[i, idxStartRealizations:]
        
        sp = ax.scatter(ParaValues,ErrValues, c=np.arange(len(ErrValues)), cmap='Oranges_r')

        minV=ParameterHistory.loc[i, 'minV']
        maxV=ParameterHistory.loc[i, 'maxV']
        
        ax.set_xlim([minV, maxV])
        ax.set_title(ParameterNames[i])
        ax.set_xlabel(ParameterNames[i])
        ax.set_ylabel('mGal (error)')
        fig.colorbar(sp)
        fig.savefig(folder+'Parameter'+str(i)+'.png', dpi = 60,bbox_inches="tight")
        plt.close('all')

## A version for a global situation
ploton = 0
idxStartRealizations = ParameterHistory.columns.tolist().index('0')
if(ploton==1):
    for i in range(nParameters):
        
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
        ParaValues=ParameterHistory.iloc[i, idxStartRealizations:]
        ErrValues = ErrorPerParameter_History[:, i]
        
        sp = ax.scatter(ParaValues,ErrValues, c=np.arange(len(ErrValues)), cmap='Oranges_r')

        minV=ParameterHistory.loc[i, 'minV']
        maxV=ParameterHistory.loc[i, 'maxV']
        
        ax.set_xlim([minV, maxV])
        ax.set_title(ParameterNames[i])
        ax.set_xlabel(ParameterNames[i])
        ax.set_ylabel('mGal (error)')
        fig.colorbar(sp)
        fig.savefig(folder+'Parameter'+str(i)+'.png', dpi = 60,bbox_inches="tight")
        plt.close('all')

ploton = 0
if(ploton==1):
    for i in range(nParameters):
        
        VarA_Vals = ParameterHistory.iloc[i, idxStartRealizations:]
        VarB_Vals = ErrorPerParameter_History[:, i]
        AllAcceptedList = AllAcceptedMatrix[:, i]
        VarA_min = ParameterHistory.loc[i, 'minV'] 
        VarA_max = ParameterHistory.loc[i, 'maxV']
        VarB_min = np.min(VarB_Vals)
        VarB_max = np.max(VarB_Vals)
        alphaVal = 0.5
        VarA_name = ParameterNames[i]
        VarB_name = 'error'
        AllNumberRunsVals = [nIterations]
        AllThreadNumber = [0]*nIterations
        Viz.MakeSearchMap(1, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, VarA_Vals, VarB_Vals, VarA_min, VarB_min,\
                  VarA_max, VarB_max, VarA_name, VarB_name, alphaVal, folder)

i=5        
j=6
VarA_Vals = ParameterHistory.iloc[i, idxStartRealizations:]
VarB_Vals = ParameterHistory.iloc[j, idxStartRealizations:]
AllAcceptedList = AllAcceptedMatrix[:, i]
VarA_min = ParameterHistory.loc[i, 'minV'] 
VarA_max = ParameterHistory.loc[i, 'maxV']
VarB_min = ParameterHistory.loc[j, 'minV'] 
VarB_max = ParameterHistory.loc[j, 'maxV']
alphaVal = 0.5
VarA_name = ParameterNames[i]
VarB_name = ParameterNames[j]
AllNumberRunsVals = [nIterations]
AllThreadNumber = [0]*nIterations
Viz.MakeSearchMap(1, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, VarA_Vals, VarB_Vals, VarA_min, VarB_min,\
          VarA_max, VarB_max, VarA_name, VarB_name, alphaVal, folder)

ploton = 0
if(ploton==1):
    for i in range(nParameters):
        
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
    
        rollingacceptance = np.cumsum(AllAcceptedMatrix[:, i])/np.arange(1, len(AllAcceptedMatrix[:, i])+1)
        ax.plot(rollingacceptance)
        
        ax.set_title(ParameterNames[i])
        ax.set_xlabel(ParameterNames[i])
        ax.set_ylabel('mGal (error)')
        fig.savefig(folder+'CumErrorParameter'+str(i)+'.png', dpi = 60,bbox_inches="tight")
        plt.close('all')

## plots that are relevant only for toy problems
ploton =0
TrueParameterValues = [2.328,2.531,2.386,2.64396,150,675,1350,1428,7142,984,4920,3015,4000,134,65,1000,3000,3000,3000]
idxStartRealizations = ParameterHistory.columns.tolist().index('0')
x = np.arange(nIterations)
if(ploton==1):
    for i in range(nParameters):
        
        fig, axs = plt.subplots(4, 1, figsize=(16,8))
        ErrValues = ErrorPerParameter_History[:, i]
        ParaValues=ParameterHistory.iloc[i, idxStartRealizations:]

        #Parameter value per iteration    
        ax=axs[0]
        
        sp = ax.plot(x, ParaValues)
        ax.plot([x[0], x[-1]], [TrueParameterValues[i],TrueParameterValues[i]],lw=2, c='k')
        minV=ParameterHistory.loc[i, 'minV']
        maxV=ParameterHistory.loc[i, 'maxV']
        
        ax.set_ylim([minV, maxV])
        ax.set_title(ParameterNames[i])
        ax.set_xlabel('iteration num')
        ax.set_ylabel(ParameterNames[i])
        
        #Parameter error per iteration    
        ax=axs[1]
        ax.plot(x, ErrValues)
        ax.set_xlabel('iteration num')
        ax.set_ylabel('Error (mGal)')

        #plot the individual acceptance stuff
        ax=axs[2]
        AllAcceptedList = AllAcceptedMatrix[:, i]
        VarA_min = ParameterHistory.loc[i, 'minV'] 
        VarA_max = ParameterHistory.loc[i, 'maxV']
        VarB_min = np.min(ErrValues)
        VarB_max = np.max(ErrValues)
        alphaVal = 0.5
        VarA_name = ParameterNames[i]
        VarB_name = 'error'
        AllNumberRunsVals = [nIterations]
        AllThreadNumber = [0]*nIterations
        Viz.MakeSearchMap(1, AllNumberRunsVals, AllAcceptedList, AllThreadNumber, ParaValues, ErrValues, VarA_min, VarB_min,\
                  VarA_max, VarB_max, VarA_name, VarB_name, alphaVal, folder, ax=ax)


        #plot the acceptance
        ax=axs[3]
        rollingacceptance = np.cumsum(AllAcceptedMatrix[:, i])/np.arange(1, len(AllAcceptedMatrix[:, i])+1)
        ax.plot(rollingacceptance)
        
        ax.set_title(ParameterNames[i])
        ax.set_xlabel(ParameterNames[i])
        ax.set_ylabel('Acceptance percentage')
        
        fig.savefig(folder+'ParameterProgress'+str(i)+'.png', dpi = 60,bbox_inches="tight")
        plt.close('all')


