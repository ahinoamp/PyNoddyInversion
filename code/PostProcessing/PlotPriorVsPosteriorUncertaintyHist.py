# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:28:59 2020

@author: ahinoamp

Plot the prior vs posterior uncertainty of parameters in histogram form
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

Parammeters2Plot = ['NewError', 'TotalError', 'Sed_Density', 'Mafic_Density', 'Mafic_MagSus', 
                    'Felsic_Density', 'Felsic_MagSus', 'Intrusive_Density', 
                    'Intrusive_MagSus', 'Sed_Thickness', 'Mafic_Thickness', 
                    'MaficFelsicCombo_Thickness', 'Tilt_Plunge Direction', 
                    'Tilt_Rotation', 'Plug0_X', 'Plug0_Y', 'Plug0_Z', 
                    'Plug0_XAxis', 'Plug0_YAxis', 'Plug0_ZAxis', 
                    'Plug0_Density', 'Plug0_MagSus', 'Plug1_X', 'Plug1_Y', 
                    'Plug1_Z',  'Plug1_XAxis', 'Plug1_YAxis', 
                    'Plug1_ZAxis', 'Plug1_Density', 'Plug1_MagSus', 'Plug2_X',
                    'Plug2_Y', 'Plug2_Z', 'Plug2_XAxis', 
                    'Plug2_YAxis', 'Plug2_ZAxis', 'Plug2_Density', 'Plug2_MagSus']

folder = ''
Data = pd.read_csv(folder+'ErrorHistorySummary.csv')
Data['Status'] = 'Prior'
PercentPosterior = 1
threshold = np.percentile(Data['TotalError'], 2.5)
filterPosterior = Data['TotalError'] < threshold
print('number of prior: ' + str(len(Data)))
print('number of posterior: ' + str(np.sum(filterPosterior)))
Data.loc[filterPosterior, 'Status'] = 'Posterior'    

for i in range(len(Parammeters2Plot)):

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(4,2))

    parameter = Parammeters2Plot[i]

    print('making plot for parameter: ' +parameter)
    datai = Data[[parameter, 'Status']]
    
    df = datai
    sns.distplot(df[parameter],  kde=True,hist=True,norm_hist=True, 
                 color="skyblue", label="Prior", ax=ax, bins=30)

    df = datai[datai['Status'] == 'Posterior']
    sns.distplot(df[parameter],  kde=True,hist=True,norm_hist=True,
                 color="red", label="Posterior", ax=ax, bins=30)
	
    # Plot formatting
    plt.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 0.95))
    plt.title('Prior vs. Posterior')
    plt.xlabel(parameter)
    plt.ylabel('Density')
    plt.xlim([np.min(datai[parameter]), np.max(datai[parameter])])
#    plt.legend()

    fig.savefig(folder+'PlotPosterior/'+parameter+'.png', dpi = 100,bbox_inches="tight")    