# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:33:36 2020

@author: ahinoamp

Create plots that analyze the results of many threads of the optimisation run 
on a cluster

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import cm

def LoadResults(folder, DataTypes):

    #nThreads = sum(os.path.isdir(folder+'/'+i) for i in os.listdir(folder))
    HyperParameters = pd.read_pickle(folder+'parameters.pkl')
  
    RanIndexList = []
    runsMismatch = []
    lengthRuns = []
    minMismatchList = []
       
    from glob import glob
    folders = glob(folder+'*/')
    nFolders = len(folders)
    
    countNot = 0
    Initialize=0
    NormalizingFactor = {}
    NormalizingFactor['Grav'] = 2.4
    NormalizingFactor['Tracer'] = 1.0
    NormalizingFactor['FaultMarkers'] = 300.
    NormalizingFactor['GT'] = 315.
    NormalizingFactor['Mag'] = 330.
    
    print(NormalizingFactor)
    Initialize=0
    DTError = {}
    #now in mismatch load a table with the non-normalized errors by 
    for i in range(nFolders):
        threadfolder = folders[i]
        ThreadNum = int(threadfolder[20:-1])
        print('reading: '+ threadfolder)
    
        try:
            mismatchPD = pd.read_csv(threadfolder+'Mismatch.csv')
        except:
            countNot= countNot+1
            continue
        
        normError = np.zeros((len(mismatchPD),))
        keys = list(mismatchPD.keys())
        Problem=False
        for dt in DataTypes:
            if(dt+'_Error' in keys):
                mismatchPD[dt] = mismatchPD[dt+'_Error'] / NormalizingFactor[dt]
            elif(dt in keys):
                norm = NormalizingFactor[dt]
                mismatchPD[dt] = mismatchPD[dt] / norm
            else:
                Problem=True
                break
            normError = normError+mismatchPD[dt]
        if(Problem):
            continue
        mismatchPD['normError'] = normError/5.0
            
        runsMismatch.append(mismatchPD['normError'].values)
        lengthRuns.append(len(mismatchPD['normError']))
             
        MinV = mismatchPD[mismatchPD['normError'] == np.min(mismatchPD['normError'])].reset_index()
    
        minMismatchList.append(MinV['normError'][0])
    
        for dt in DataTypes:
            if(Initialize==0):
                DTError[dt] = [MinV[dt][0]]
            else:
                DTError[dt].append(MinV[dt][0])
        Initialize=1
        
       
        RanIndexList.append(ThreadNum)
    
    sV = np.argsort(np.asarray(RanIndexList))
    RanIndexList = np.asarray(RanIndexList)[sV]
    minMismatchList = np.asarray(minMismatchList)[sV]
    for dt in DataTypes:
        DTError[dt]=np.asarray(DTError[dt])[sV]
        
    HyperParameters=HyperParameters[HyperParameters['thread_num'].isin(RanIndexList)].reset_index(drop=True)
    HyperParameters['MinMismatch']=minMismatchList
    for dt in DataTypes:
        HyperParameters[dt]=DTError[dt]
    
    HyperParameters.to_csv(folder+'ParametersWithMismatch.csv')

    MismatchMatrix = np.zeros((np.max(lengthRuns),len(RanIndexList)))*np.NaN
    for i in range(len(runsMismatch)):
        MismatchMatrix[0:len(runsMismatch[i]),i] = runsMismatch[i]
    
    return HyperParameters, MismatchMatrix

folder= 'Combo_Scratch/'
DataTypes = ['Grav','Mag','Tracer','GT','FaultIntersection']

HyperParameters, MismatchMatrix = LoadResults(folder, DataTypes)

##########################################
## Clean up quant vs. categorical variables
##########################################
NumbersTreatCategorical= ['OptimMethod', 'ScenarioNum', 'GeneralPerturbStrategy',  
                  'SimulationShiftType', 'cubesize',
                  'AcceptProbType', 'ErrorType', 'LearningRate', 'LocalWeightsMethod', 
                  'ErrorNorm', 'errCalcMethodFaultIntersection', 'LocalWeightsMode',
                  'SelectionMethod', 'MatingMethodLocal','MatingMethodGlobal', 'MO_WeightingMethod']

NotHyperParameters= ['Grav', 'Mag', 'Tracer', 'GT', 'FaultIntersection', 'MinMismatch', 'thread_num',
                     'Toy', 'verbose', 'OutputImageFreq', 'BaseFolder']
QuantVariables = []
CatagoricalVars = []

parameters = list(HyperParameters.columns)
dtypes = list(HyperParameters.dtypes)
for i in range(len(parameters)):
    parameter = str(parameters[i])
    dtype_i = dtypes[i].name

    if(parameter in NotHyperParameters):
        continue

    #No need for analysis if the categorical variable is unique
    filterNaN = ~pd.isnull(HyperParameters[parameter])
    nonNanVals = HyperParameters.loc[filterNaN, parameter]
    nUniqueCats = len(np.unique(nonNanVals))
    if(nUniqueCats<2):
        continue

    if((dtype_i!='object') & (parameter not in NumbersTreatCategorical)):
        QuantVariables.append(parameter)
    else:
        CatagoricalVars.append(parameter)
        
##########################################
## Plot error histogram
##########################################
minMismatchList = HyperParameters['MinMismatch']

fig, axs = plt.subplots(1, 6, figsize=(15,3))
ax = axs[0]
ax.hist(minMismatchList, bins=20)
ax.set_xlabel('Error')
ax.set_ylabel('Number')
ax.set_title('Combo Error ' +'(' +'{:.3f}'.format(np.mean(minMismatchList)) +')')
i=0
for dt in DataTypes:
    ax = axs[i+1]
    ax.hist(HyperParameters[dt], bins=20)
    ax.set_xlabel('Error')
    ax.set_ylabel('Number')
    ax.set_title('Error for '+dt + ' (' +'{:.3f}'.format(np.mean(HyperParameters[dt])) +')')
    i=i+1

plt.savefig(folder+'ErrorHistogram_'+dt+'.png', dpi=150,bbox_inches='tight')

##########################################
## Plot error progression by Optim method
##########################################

fig, axs = plt.subplots(1, 2, figsize=(10,4))

filterMCMC = (HyperParameters['OptimMethod']=='MCMC')
filterNotMCMC = ~(HyperParameters['OptimMethod']=='MCMC')

#plot MCMC
dataMCMC=MismatchMatrix[:, filterMCMC]
axs[0].plot(dataMCMC)

#plot Not MCMC
dataNotMCMC=MismatchMatrix[:, filterNotMCMC]
axs[1].plot(dataNotMCMC)
  
axs[0].set_title('Mismatch - MCMC')
axs[1].set_title('Mismatch - Swarm methods')

fig.savefig(folder+'MismatchPerIteration.png',dpi=150,bbox_inches='tight')
plt.close(fig)

##########################################
## Plot error progression by categorical variables
##########################################
     
for cat in CatagoricalVars:    
    print('making plot generations for '+cat)
    CatValues = HyperParameters[cat]
    uniqueCat = np.unique(CatValues[~pd.isnull(CatValues)])
    Colors = cm.rainbow(np.linspace(0,1,len(uniqueCat)))

    filterMCMC = (HyperParameters['OptimMethod']=='MCMC')
    filterNotMCMC = ~(filterMCMC)

    fig, axs = plt.subplots(1, 2, figsize=(10,4))
    for i in range(len(uniqueCat)):
        colCatFilter = CatValues==uniqueCat[i]        

        #plot MCMC
        dataMCMC=MismatchMatrix[:, colCatFilter&filterMCMC]
        Mean = np.nanmean(dataMCMC, axis=1)
        Upper = np.nanpercentile(dataMCMC, 90, axis=1)
        Lower = np.nanpercentile(dataMCMC, 10, axis=1)
        Iterations = np.arange(np.shape(dataMCMC)[0])
        axs[0].fill_between(Iterations, Lower, Upper, color=Colors[i], alpha=.2)
        axs[0].plot(Mean, color=Colors[i], alpha=1, label=uniqueCat[i])

        #plot Not MCMC
        dataNotMCMC=MismatchMatrix[:, colCatFilter&filterNotMCMC]
        Mean = np.nanmean(dataNotMCMC, axis=1)
        Upper = np.nanpercentile(dataNotMCMC, 90, axis=1)
        Lower = np.nanpercentile(dataNotMCMC, 10, axis=1)
        Iterations = np.arange(np.shape(dataNotMCMC)[0])
        axs[1].fill_between(Iterations, Lower, Upper, color=Colors[i], alpha=.2)
        axs[1].plot(Mean, color=Colors[i], alpha=1, label=uniqueCat[i])

    axs[0].legend()
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Error (MCMC)')
    axs[0].set_title('Error per iteration split by ' +cat)

    axs[1].legend()
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Error (Not MCMC)')
    axs[1].set_title('Error per iteration split by ' +cat)
    
    plt.savefig(folder+'ErrorIterationCat_'+cat+'.png',dpi=150,bbox_inches='tight')
    plt.close('all')  


##########################################
## Plot error per interaction of categorical variables
##########################################

for cat in CatagoricalVars:
    print('Making cat plot for ' + cat)

    chart = sns.catplot(x=cat, y="MinMismatch", kind='box', data=HyperParameters);
    chart.set_xticklabels(rotation=45, horizontalalignment='right')
    plt.savefig(folder+'Cat_'+cat+'.png',dpi=150,bbox_inches='tight')
    plt.close('all')  
    
##########################################
## Plot error per interaction of categorical variables
##########################################

for i in range(len(CatagoricalVars)):

    catA = CatagoricalVars[i]  

    for j in range(len(CatagoricalVars)):
        if(j<=i):
            continue

        catB = CatagoricalVars[j]

        print('Making cat plot for ' + catA + ' with ' + catB)

        sns.catplot(x=catA, y="MinMismatch", hue=catB, kind="box", data=HyperParameters);
        plt.savefig(folder+'CatInteraction_'+catA+'_'+catB+'.png',dpi=150,bbox_inches='tight')
        plt.close('all')  
                
##########################################
## Plot error for quant variables
##########################################

for quant in QuantVariables:
    print('Making quant plot for ' + quant)

    fig, ax = plt.subplots(1, 1, figsize=(10,4))
    quantV = HyperParameters[quant].values
    mismatch = HyperParameters['MinMismatch'].values

    filterV = ~(pd.isnull(quantV) |pd.isnull(mismatch))  
    if(sum(filterV)==0):
        continue
    quantV = quantV[filterV]
    mismatch = mismatch[filterV]

    ax.scatter(quantV,mismatch)
    plt.plot(np.unique(quantV), np.poly1d(np.polyfit(quantV, mismatch, 1))(np.unique(quantV)), color='k')
    cc = np.corrcoef(quantV, mismatch)
    if(np.any(np.isnan(cc))):
        print(quantV)
        print(mismatch)
        print(mismatch.dtype)
        print(quantV.dtype)
    ax.set_xlabel(quant)
    ax.set_ylabel('mismatch')
    ax.set_title('Correlation coeff: ' + '{:.5f}'.format(cc[0,1]))
    plt.savefig(folder+'MismatchVsQuant_'+quant+'.png',dpi=150,bbox_inches='tight')
    plt.close('all')