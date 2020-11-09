# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 10:04:17 2020

@author: ahinoamp
"""
import numpy as np
import David_falsification as dfalse
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

DataTypesList = [['grav', 'mag', 'gt'], ['grav'], ['mag'], ['gt']]
Names= ['allThree', 'grav','mag', 'gt']
nCList = [40, 30, 30,30]
# load the data
# realizatons x data_samples x data_types.
folder = ''

for d in range(4):
    run=1
    DataTypes = DataTypesList[d]
    Name = Names[d]
    if(run==1):
        for i in range(len(DataTypes)):
            DataType = DataTypes[i]
            data_iObs = np.loadtxt(folder+DataType+'_obs.csv', delimiter=',')
            data_iObs = data_iObs.reshape(1,-1)
            data_iSim = np.loadtxt(folder+DataType+'_largeMatrix.csv', delimiter=',').T
            meanV = np.mean(data_iSim)
            stdV = np.std(data_iSim)
            data_iSim = (data_iSim-meanV)/stdV
            data_iObs = (data_iObs-meanV)/stdV
            data_i = np.concatenate((data_iObs, data_iSim))        
            
            truncSVD = decomposition.TruncatedSVD(n_components=31).fit(data_i)
            TwoComp = truncSVD.transform(data_i)
            variance = truncSVD.explained_variance_ratio_
            explainedVariance = np.sum(variance)
            print(explainedVariance)
            
            if(i==0):
                data_mat = TwoComp
            else:
                data_mat = np.concatenate((data_mat, TwoComp), axis=1)
        np.savetxt(folder+'allSimData_'+Name+'.csv', data_mat, delimiter=',', fmt='%.4f') 
    
    
    DataAll = np.loadtxt(folder+'allSimData_'+Name+'.csv',delimiter=',')
    # reduce to one file
    #d_obs_mat = 2
    #scalar=1
    
    nC=nCList[d]
    truncSVD = decomposition.TruncatedSVD(n_components=nC).fit(DataAll)
    TwoComp = truncSVD.transform(DataAll)
    variance = truncSVD.explained_variance_ratio_
    print(np.sum(variance))
    
    #dmat_4mpca, dobsmat_4mpca = mixPCA.dmat_4mixpca(data_mat, d_obs_mat, scalar)
    
    # divide by first eigenvalue
    print('analysis starting')
    d_var = TwoComp 
    d_obs = TwoComp[0, :].reshape(-1,nC)
    prior_name='prior name'
    plt_OrNot=True
    Q_quantile = 97.5
    dfalse.RobustMD_flsification(d_var, d_obs, prior_name, plt_OrNot, Q_quantile, Name, variance)
    
    # move into reduced space
    
    
    # do mahalanobis
    
    
    # plot
