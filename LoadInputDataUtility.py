# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:45:19 2020

@author: ahinoamp
"""
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

def loadData(P):

    #Gravity observations
    ########################
    ObsData = pd.read_csv('Data/GravityData.csv')
    filterLimits =  ((ObsData['X']>P['xmin']) & (ObsData['Y']>P['ymin'])   
                     & (ObsData['X']<(P['xmax'])) 
                     & (ObsData['Y']<(P['ymax']))) 
    ObsData = ObsData[filterLimits]
    
    P['Grav'] = {}
    
    P['Grav']['xObs'] = ObsData['X'].values
    P['Grav']['yObs'] = ObsData['Y'].values
    P['Grav']['Obs'] = ObsData['CBA_240'].values
    P['Grav']['medianObs'] = np.median(P['Grav']['Obs'])
    P['Grav']['stdObs'] = np.std(P['Grav']['Obs'])
    P['Grav']['nObsPoints'] = len(P['Grav']['Obs'])

    #Granite top observations
    P['GT'] = {}
    GraniteTopObs = pd.read_csv('Data/GraniteTop.csv')
    P['GT']['xObs'] = GraniteTopObs['X'].values
    P['GT']['yObs'] = GraniteTopObs['Y'].values
    P['GT']['Obs'] = GraniteTopObs['Z'].values
    P['GT']['nObsPoints'] = len(P['GT']['Obs'])

    #Magnetics observations
    P['Mag'] = {}
    MagneticsObs = pd.read_csv('Data/Magnetics.csv')
    filterLimits =  ((MagneticsObs['X']>P['xmin']) 
                     & (MagneticsObs['Y']>P['ymin'])   
                     & (MagneticsObs['X']<(P['xmax'])) 
                     & (MagneticsObs['Y']<(P['ymax']))) 
    MagneticsObs = MagneticsObs[filterLimits]

    P['Mag']['xObs'] = MagneticsObs['X'].values
    P['Mag']['yObs'] = MagneticsObs['Y'].values
    P['Mag']['Obs'] = MagneticsObs['VALUE'].values
    P['Mag']['nObsPoints'] = len(P['Mag']['Obs'])
    P['Mag']['medianObs'] = np.median(P['Mag']['Obs'])

    #Tracer data
    #make sure the well paths only include those sections below the casing
    P['Tracer'] = {}
    WellPathsOrig = pd.read_csv('Data/AllWellPathsData.csv')
    WellPaths = WellPathsOrig.copy()
    CasingShoe = pd.read_csv('Data/CasingShoe.csv')
    for i in range(len(CasingShoe)):
        well = CasingShoe.loc[i, 'WellName']
        casingZ = CasingShoe.loc[i, 'Z']
        removeV = (WellPaths['WellName']==well)&(WellPaths['Zm']>=casingZ) 
        WellPaths = WellPaths[~removeV]
    P['WellPaths'] = WellPaths

    TracerConnections = pd.read_csv('Data/TracerConnections.csv')
    RowsMaxZ = WellPathsOrig[WellPathsOrig['Zm'] == WellPathsOrig.groupby('WellName')['Zm'].transform('max')]
    x1, y1, x2, y2 = [], [],[],[]
    for i in range(len(TracerConnections)):
        filterI = RowsMaxZ['WellName']==TracerConnections.loc[i, 'Injector']
        x1.append(RowsMaxZ.loc[filterI,'Xm'].values[0])
        y1.append(RowsMaxZ.loc[filterI,'Ym'].values[0])
        filterP = RowsMaxZ['WellName']==TracerConnections.loc[i, 'Producer']
        x2.append(RowsMaxZ.loc[filterP,'Xm'].values[0])
        y2.append(RowsMaxZ.loc[filterP,'Ym'].values[0])
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    TracerConnections['x1'] = x1
    TracerConnections['x2'] = x2
    TracerConnections['y1'] = y1
    TracerConnections['y2'] = y2
    TracerConnections['dx'] = x2-x1
    TracerConnections['dy'] = y2-y1
    TracerConnections['midx'] = (x2+x1)/2
    TracerConnections['midy'] = (y2+y1)/2

    P['Tracer']['Connections']=TracerConnections
    P['Tracer']['nObsPoints'] = len(P['Tracer']['Connections'])
    P['Tracer']['xObs'] = (x1+x2)/2
    P['Tracer']['yObs'] = (y1+y2)/2
    
    # Fault intersection data
    P['FaultMarkers'] = {}
    FaultMarkers= pd.read_csv('Data/MarkersInversion.csv')
    FaultMarkers['wellid'] = FaultMarkers.groupby(['WellName']).ngroup()
    P['FaultMarkers']['Obs'] = FaultMarkers
    Wellnames = WellPathsOrig['WellName']
    FaultMarkerWells = np.unique(P['FaultMarkers']['Obs']['WellName'])
    filterWells = np.isin(Wellnames, FaultMarkerWells)
    WellPathsTracers = WellPathsOrig[filterWells]
    nWells=len(FaultMarkerWells)
    zWells = np.zeros((nWells,2))
    idWells = np.zeros((nWells,2))
    topXWell = np.zeros((nWells,))
    topYWell = np.zeros((nWells,))
    idPlotWell = np.zeros((nWells,), dtype=int)

    WellsAtMaxZ = WellPathsTracers[WellPathsTracers['Zm'] == WellPathsTracers.groupby('WellName')['Zm'].transform('max')]

    for i in range(nWells):
        filterWell = WellPathsTracers['WellName']==FaultMarkerWells[i]
        z = WellPathsTracers.loc[filterWell, 'Zm']
        zWells[i, 0] = np.min(z)
        zWells[i, 1] = np.max(z)
        idWells[i,:] = i
 
        filterMaxTable = WellsAtMaxZ['WellName']==FaultMarkerWells[i]
        topXWell[i] = WellsAtMaxZ.loc[filterMaxTable, 'Xm'].values[0]
        topYWell[i] = WellsAtMaxZ.loc[filterMaxTable, 'Ym'].values[0]
        idPlotWell[i] = i
        
    P['FaultMarkers']['WellData'] =  WellPathsTracers[WellPathsTracers['Zm']<1200].copy(deep=True)
    P['FaultMarkers']['WellData']['id'] = P['FaultMarkers']['WellData'].groupby(['WellName']).ngroup()

    P['zWells'] = zWells.T
    P['idWells'] = idWells.T
    P['topXWell'] = topXWell.T
    P['topYWell'] = topYWell.T
    P['idPlotWell'] = idPlotWell.T

    P['FaultMarkers']['nObsPoints'] = len(P['FaultMarkers']['Obs'])
    P['FaultMarkers']['xObs'] = P['FaultMarkers']['Obs']['X']
    P['FaultMarkers']['yObs'] = P['FaultMarkers']['Obs']['Y']