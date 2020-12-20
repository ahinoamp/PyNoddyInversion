# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:28:59 2020

@author: ahinoamp
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


folder = 'C:/Users/ahino/Documents/FinalGeothermicsToday/'
folderresults = os.path.abspath(os.path.join(os.path.dirname( __file__), '..', 'Combo_Scratch'))
file=folder+'/'+'ErrorHistorySummary.csv'
Data = pd.read_csv(file)
Data['Status'] = 'Prior'

P={}
P['Grav_comboNormalizingFactor'] = 2.4
P['Tracer_comboNormalizingFactor'] = 1.0
P['FaultMarkers_comboNormalizingFactor'] = 500
P['GT_comboNormalizingFactor'] = 315
P['Mag_comboNormalizingFactor'] = 300
            
DataTypes = ['Grav', 'Tracer', 'FaultMarkers', 'GT', 'Mag']
RealError = []
for i in range(len(Data)):
    Err = 0
    for dt in DataTypes:
        Err=Err+Data.loc[i, dt+'_Error']/P[dt+'_comboNormalizingFactor']

    RealError.append(Err/5.)
        
Data['RealError'] = RealError
          
PercentPosterior = 1
threshold = np.percentile(Data['RealError'], 1.5)
filterPosterior = Data['RealError'] < threshold


datakeys = list(Data.keys())
Parammeters2Plot = []
for i in range(len(datakeys)):
    if(('Fault' not in datakeys[i])&('filename' not in datakeys[i])&('Status' not in datakeys[i])):
#    if(('filename' not in datakeys[i])&('Status' not in datakeys[i])):
        Parammeters2Plot.append(datakeys[i])
        
print('number of prior: ' + str(len(Data)))
print('number of posterior: ' + str(np.sum(filterPosterior)))
Data.loc[filterPosterior, 'Status'] = 'Posterior'    

for i in range(len(Parammeters2Plot)):

    plt.close('all')

    fig, ax = plt.subplots(1, 1, figsize=(4,2))

    parameter = Parammeters2Plot[i]

    print('making plot for parameter: ' +parameter)
    datai = Data[[parameter, 'Status']]
 
    if('MagSus' in parameter):
        df = datai
        sns.distplot(np.log10(df[parameter]),  kde=True,hist=True,norm_hist=True, 
                      color="skyblue", label="Prior", ax=ax, bins=30)
    
        df = datai[datai['Status'] == 'Posterior']
        sns.distplot(np.log10(df[parameter]),  kde=True,hist=True,norm_hist=True,
                      color="red", label="Posterior", ax=ax, bins=30)
        plt.xlim([np.log10(np.min(datai[parameter])), np.log10(np.max(datai[parameter]))])
    else:
        df = datai
        sns.distplot(df[parameter],  kde=True,hist=True,norm_hist=True, 
                      color="skyblue", label="Prior", ax=ax, bins=30)
    
        df = datai[datai['Status'] == 'Posterior']
        sns.distplot(df[parameter],  kde=True,hist=True,norm_hist=True,
                      color="red", label="Posterior", ax=ax, bins=30)
        plt.xlim([np.min(datai[parameter]), np.max(datai[parameter])])
        
        # Plot formatting
    plt.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 0.95))
    plt.title(parameter + ': Prior vs. Posterior')
    plt.xlabel(parameter)
    plt.ylabel('Density')
#    plt.legend()

    fig.savefig('C:/Users/ahino/Documents/FinalGeothermicsToday/Plots/'+parameter+'.png', dpi = 125,bbox_inches="tight") 
    
FaultProperties = ['Amplitude', 'Dip', 'Dip Direction', 'Pitch', 'Profile Pitch', 'XAxis', 'YAxis', 'ZAxis', 'X', 'Y', 'Z', 'Slip']
Zones = ['MidEast','MidMid', 'MidWest','NorthEast',
         'NorthMid','NorthWest', 'SouthEast', 'SouthMid','SouthWest']

ColumnNames = list(Data.keys())

for fp in range(len(FaultProperties)):

    prop = FaultProperties[fp]

    for z in range(len(Zones)):

        plt.close('all')

        zone = Zones[z]
        fig, ax = plt.subplots(1, 1, figsize=(4,2))
        pname = 'FaultZone_'+zone+'Prop_'+prop

        print('making plot for: '+pname)

        subseti = [] 
        for c in range(len(ColumnNames)):
            cn = ColumnNames[c]
            if(prop == 'Dip'):
                if (('Fault' in cn) and (prop in cn) and (zone in cn) and('Dip Direction' not in cn)):
                    subseti.append(cn)
            else:
                if (('Fault' in cn) and (prop in cn) and (zone in cn)):
                    subseti.append(cn)
    
        datai = Data[subseti+['Status']]
    
        if(len(datai)==0):
            continue
        
        df = datai
        sns.distplot(df[subseti][:],  kde=True,hist=True,norm_hist=True, 
                 color="skyblue", label="Prior", ax=ax, bins=30)

        df = datai[datai['Status'] == 'Posterior']
        sns.distplot(df[subseti][:],  kde=True,hist=True,norm_hist=True,
                   color="red", label="Posterior", ax=ax, bins=30)            
    # Plot formatting
        plt.legend(prop={'size': 12}, loc='upper left', bbox_to_anchor=(1, 0.95))
        plt.title('Prior vs. Posterior')
        plt.xlabel(pname)
        plt.ylabel('Density')
#        plt.xlim([np.min(datai[parameter]), np.max(datai[parameter])])
        fig.savefig('C:/Users/ahino/Documents/FinalGeothermicsToday/Plots/'+pname+'.png', dpi = 100,bbox_inches="tight") 

        