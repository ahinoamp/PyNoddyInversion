# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 12:22:14 2020

@author: ahinoamp
"""

from sklearn.covariance import MinCovDet as MCD

from scipy import stats

import numpy as np

import matplotlib.pyplot as plt

def RobustMD_flsification(d_var, d_obs, prior_name, plt_OrNot, Q_quantile,
                          Name, variance):
 
    '''

    This function falsifies the prior using Robust Mahalanobis Distance RMD. 

    d_var: the data variable, 2D array, (n_realizations X p_features)

    d_obs: the data observation variable, 2D array, (1xp)

    prior_name: name of the prior model for falsification, string

    plt_OrNot: True or False, to create the distribution plot of the calculated RMDs.

    Q_quantile: the Q_quantile of the RMD distribution, 95 or 97.5 is suggested

    example: MD_flsification(d_pri, d_obs, True, 95) will produce the RMD_obs, RMD_pri, RMD_Q95, and plot them.

    '''

   

    mcd = MCD(random_state=0).fit(d_var)

    new_obs = d_obs-mcd.location_

    md_obs= np.sqrt(new_obs.dot(np.linalg.inv(mcd.covariance_)).dot(new_obs.T))

    print('Robust Mahalanobis Distance of d_obs = ', md_obs[0,0].round(decimals = 3))

    md_samples=[]

    for i in range(len(d_var)):

        sample = d_var[i:i+1, :]-mcd.location_

        md_samp = np.sqrt(sample.dot(np.linalg.inv(mcd.covariance_)).dot(sample.T))[0,0]

        md_samples.append(md_samp)

    md_samples = np.asarray(md_samples)

    print(str(Q_quantile)+'th Quantile of Robust Mahalanobis Distance is', \

          stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3))

 

    if plt_OrNot == True:

        fig, axs = plt.subplots(1, 2, figsize=(10,4))
        fz=11
        ax=axs[0]
        ax.scatter(d_var[:,0], d_var[:,1], c=abs(md_samples), 
                    cmap ='winter_r', s=10, vmax = md_samples.max(), vmin=md_samples.min(),
                    linewidths=0.5, edgecolor='k', label='Simulated')        
        ax.scatter(d_obs[:, 0], d_obs[:, 1], c=md_obs[0], cmap ='winter_r', marker='D', s=110, 
            vmax = md_samples.max(), vmin=md_samples.min(),
            linewidths=3, edgecolor='red', label='Observed')
        ax.set_ylabel('PC2 (' + "{:.1f}".format(variance[0]*100)+'% explained variance)', fontsize=fz)
        ax.set_xlabel('PC1 (' + "{:.1f}".format(variance[1]*100)+'% explained variance)', fontsize=fz)
        ax.set_title('(a) Simulated vs. observed data in PC space', 
                  fontsize=fz)
        ax.legend()
                
        ax=axs[1]
        hmm = ax.scatter(np.arange(1,(len(d_var)+1)), md_samples, c=abs(md_samples), 
                    cmap ='winter_r', s=10, vmax = md_samples.max(), vmin=md_samples.min(),
                    linewidths=0.5, edgecolor='k')

        ax.scatter([0], md_obs, c=md_obs, cmap ='winter_r', marker='D', s=110, 
                    vmax = md_samples.max(), vmin=md_samples.min(),
                    linewidths=3, edgecolor='red')

        ax.set_ylabel('Robust mahalanobis distance', fontsize=fz)

        ax.set_xlabel('Realization number', fontsize=fz)

        ax.set_xlim(-8, len(md_samples))

        ax.hlines(y=stats.scoreatpercentile(md_samples, Q_quantile), xmin= -10, 
                   xmax=len(md_samples), colors='red', linewidths=2, linestyles='--')

        cbar = fig.colorbar(hmm, fraction=0.035, ax=ax)

        cbar.ax.set_ylabel('RMD')

        ax.set_title('(b) Prior falsification', 
                  fontsize=fz)

        fig.savefig('Priorfalse_'+Name+'.png', dpi = 300,bbox_inches="tight")   

    return md_obs[0,0].round(decimals = 3), stats.scoreatpercentile(md_samples, Q_quantile).round(decimals=3)