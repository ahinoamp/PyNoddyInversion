# -*- coding: utf-8 -*-
"""
Created on September 23, 2020

@author: ahinoamp@gmail.com

This script provides general functions necessary for inversion:
    1. Initialization functions: such as setting up the model paramters table
    2. Weights related functions
    3. Output functions
    4. Error access functions
    5. Check whether to terminate function
"""
import numpy as np
import random
import pandas as pd
from pathlib import Path
from shutil import copyfile

import VisualizationUtilities as Viz
import PriorUncertaintyUtil as Unc
import SamplingHisFileUtil as sample
import SimulationUtilities as sim

##########################################
## Initialization functions
##########################################
def setupFoldersParameters(P):    
    '''Ranodm logistics for running the simulations'''

    # Crete necessary output folders
    folder = P['HypP']['BaseFolder']+'/Thread'+str(P['HypP']['thread_num'])+'/'
    P['folder']=folder
    Path(folder).mkdir(parents=True, exist_ok=True)
    folderHis = folder+'HistoryFileInspection/'
    Path(folderHis).mkdir(parents=True, exist_ok=True)
    folderViz = folder+'VisualInspection/'
    Path(folderViz).mkdir(parents=True, exist_ok=True)

    # Define output folder name
    P['SampledInputFileName'] = folder+'scenario_scratch.his'
    P['output_name'] = folder+'noddy_out'
    
    #If optimizine one at a time, then determine the datatype start index
    P['StartIdx'] = random.choice([0,1,2,3,4])

    # Define the model limits
    P['xmin'] = P['HypP']['xy_origin'][0]
    P['xmax'] = P['HypP']['xy_origin'][0]+P['HypP']['xy_extent'][0]

    P['ymin'] = P['HypP']['xy_origin'][1]
    P['ymax'] = P['HypP']['xy_origin'][1]+P['HypP']['xy_extent'][1]

    P['zmin'] = P['HypP']['xy_origin'][2]
    P['zmax'] = P['HypP']['xy_origin'][2]+P['HypP']['xy_extent'][2]
    
    # Until the normalization factor is calculated, can set an initial guess
    # of the data normalization factor
    if(P['HypP']['DatNormMethod']=='MedianInitialRounds'):
        P['DatNormCoef'] = {}
        for dt in P['HypP']['DataTypes']:
            P['DatNormCoef'][dt]=1
    else:
        P['DatNormCoef'] = P['HypP']['DatNormCoef']

    # Transfer some hyper parameters into the main parameters dictionary for 
    # easy access
    P['nRuns'] =int(P['HypP']['nruns'])
    P['verbose'] = P['HypP']['verbose']
    P['cubesize'] = P['HypP']['cubesize']
    P['DataTypes'] = P['HypP']['DataTypes']


def InitializeParameters(P):
    '''Set up the model parameters table based on the hyper parameters and 
    random choices'''
    
    # Define the range of uncertainties for all possible faulting scenarios
    ModelParamTable= Unc.DefinePriorUncertainty(P)   

    # choose only a subset of faults from the fault bank
    ModelParamTable = Unc.SelectFaultingEvents(ModelParamTable, P)        
    ModelParamTable = Unc.OrderFaultingEvents(ModelParamTable,P)        

    # Create a history file template that will later be used to update with 
    # new parameters    
    sample.CreatePyNoddyTemplateGivenParameterTablePatua(P, ModelParamTable)

    # Create an object which tracks all of the Noddy parameters, not only those
    # that are optimized
    P['FullParamHis'] = ModelParamTable.copy()

    # only keep those parameters that need to be optimized in the model 
    # parameter table
    P['OptimizeParametersIdx'] = ModelParamTable['std']>0.00001
    ModelParamTable=ModelParamTable[ModelParamTable['std']>0.00001].reset_index()
    ModelParamTable = ModelParamTable.drop(['index'], axis=1)

    # Book keeping
    P['nParam']= np.shape(ModelParamTable)[0]
    P['ModelParamTable']=ModelParamTable
    P['nEvents']=int(np.max(ModelParamTable['EventNum']))
    P['nFaults']=int(np.sum(ModelParamTable['EventName'].drop_duplicates().str.contains('Fault')))
    
    return ModelParamTable

def register_sim_functions(P, toolbox):
    '''Create a toolbox of functions
       This way, the function always has the same name, but can have 
       different content. Allows to keep the same algorithm structure, but also
       test some interesting variations.'''
    # gravity shift type
    if(P['SimulationShiftType'] == 'Median Datum Shift'):
        toolbox.register("shift_geophys", sim.median_shift)
    elif(P['SimulationShiftType'] == 'Constant Datum Shift'):
        toolbox.register("shift_geophys", sim.const_shift)
    elif(P['SimulationShiftType'] == 'Median Const Datum Shift'):
        toolbox.register("shift_geophys", sim.median_const_shift)
    else:
        toolbox.register("shift_geophys", sim.no_shift)
        
##########################################
## Weights functions
#     local weights: weights for each data point when calculating the overall 
#                    error per parameters, data points near the parameter 
#                    will have higher weights
#     data normaliziation: coefficients that should make the different errors
#                          range between one and zero
#     data optimization: giving different weights to different data types 
#                        during the optimization
##########################################


def CalcDatNormCoef(P):
    '''Calculate the normalization factor by taking the median error of the 
    initial exploration rounds'''

    #take all of the measured errors for each dataType
    for dt in P['DataTypes']:
        P['DatNormCoef'][dt] = np.nanmedian(P[dt]['L1MismatchList'])
        if(P['verbose']):
            print('Norm Factor: ' +dt+ ': '+str(P['DatNormCoef'][dt]))


def UpdateLocalWeights(P):
    '''Calculate the weight of each data point point for calculating the error
    attributed to a Param'''
    
    if((P['HypP']['LocalWeightsMode']=='Once') & (P['iterationNum']==0)):
        for dt in P['DataTypes']:      
             P[dt]['LocalErrorWeights'] = CalcLocalErrWts_MinMaxRanges(P, dt)
    elif(P['HypP']['LocalWeightsMode']=='Many'):
        if(np.mod(P['iterationNum'],P['HypP']['UpdateLocalWeightsFreqRuleBased'])==0):
            for dt in P['DataTypes']:      
                 P[dt]['LocalErrorWeights'] = CalcLocalErrWts(P, dt)            
          
def CalcLocalErrWts(P, dt):
    '''Determine which data points contribute to the error of a Param'''
    
    #Coordinates of the observed data in model space
    xObs= P[dt]['xObs']-P['xmin']
    yObs= P[dt]['yObs']-P['ymin']
    
    FullParamHis = P['FullParamHis']
    CurrValsFullParamHis =FullParamHis.iloc[:, -1]
    ModelParamTable = P['ModelParamTable']
    
    # Initialize matrix for the weights
    WeightsMatrix = np.zeros((P[dt]['nObsPoints'], P['nParam']))
    
    # Loop through the events and assign weights to data points based on an 
    # approximation of the area of influence of the event
    EventNames = ModelParamTable['EventName'].values
    Events = np.unique(ModelParamTable['EventNum'])
    for i in Events:
        # Find the Params associated with this event
        eventIndices = ModelParamTable['EventNum'].values==i
        eventName = EventNames[eventIndices][-1]

        #if the event is tilt or stratigraphy, then all of the observations 
        #equally contribute to the calculated error
        if(('Tilt' in eventName)| (i==1)):
            WeightsMatrix[:, eventIndices] = 1/ P[dt]['nObsPoints']

        # if the event is a fault, then observations points within an ellipse
        # around the fault contribute to the total error attributed to the 
        # Param
        elif('Fault' in eventName):
            props = ['X', 'Y', 'YAxis', 'ZAxis', 'Dip Direction']
            eventFilter = FullParamHis['EventNum'].values==i
            propV = {}
            for prop in props:
                idx = (eventFilter & (FullParamHis['Prop']==prop))
                propV[prop]= CurrValsFullParamHis.loc[idx].values[-1]
            
            tX = xObs-propV['X']
            tY = yObs-propV['Y']
            a = (propV['YAxis']/2)*P['HypP']['localWeightRadiusMult']
            b = (propV['ZAxis']/2)*P['HypP']['localWeightRadiusMult']
            alpha = np.deg2rad(-(propV['Dip Direction']+90))
            part1 = np.power(tX*np.cos(alpha)+tY*np.sin(alpha), 2)/a**2
            part2 = np.power(tX*np.sin(alpha)-tY*np.cos(alpha), 2)/b**2
            condition1 = (part1+part2) < 1
            WeightsMatrix[np.ix_(condition1, eventIndices)] = 1
        # if the event is an intrusion
        else:
            props = ['X', 'Y', 'YAxis', 'XAxis']
            eventFilter = FullParamHis['EventNum'].values==i
            propV = {}
            for prop in props:
                idx = (eventFilter & (FullParamHis['Prop']==prop))
                propV[prop]= CurrValsFullParamHis.loc[idx].values[-1]

            tX = xObs-propV['X']
            tY = yObs-propV['Y']
            Ry = propV['YAxis']*P['HypP']['localWeightRadiusMult']
            Rx = propV['XAxis']*P['HypP']['localWeightRadiusMult']
            R = np.max([Ry, Rx])
            condition1 = (np.power(tX, 2)+np.power(tY,2)) < (R**2)
            WeightsMatrix[np.ix_(condition1, eventIndices)] = 1

    #sometimes there can be errors in the data. To avoid that problem, we give those guys zeros weight
    WeightsMatrix[np.isnan(WeightsMatrix)] = 0
    sumWeightsPerParam = np.sum(WeightsMatrix, axis=0)
    filterSumZero = sumWeightsPerParam==0
    WeightsMatrix[:, ~filterSumZero] = WeightsMatrix[:, ~filterSumZero]/sumWeightsPerParam[~filterSumZero]
    WeightsMatrix[:, filterSumZero] = 1/ P[dt]['nObsPoints']

    return WeightsMatrix

def CalcLocalErrWts_MinMaxRanges(P, dt):
    '''Calculate the area of influence of parameters by scanning the minimum 
    and maximum values they can take'''

    xObs= P[dt]['xObs']-P['xmin']
    yObs= P[dt]['yObs']-P['ymin']

    FullParamHis = P['FullParamHis']
    
    ModelParamTable = P['ModelParamTable']
    ParamsNames = ModelParamTable['EventName']+'_'+ModelParamTable['Prop']
    ParamsNames = ParamsNames.values.tolist()
    EventNames = ModelParamTable['EventName'].values
    EventNum = ModelParamTable['EventNum'].values
    Events = np.unique(EventNum)
    
    WeightsMatrix = np.zeros(( P[dt]['nObsPoints'], P['nParam']))
           
    for i in Events:
        eventIndices = EventNum==i
        eventName = EventNames[eventIndices][-1]
        if(('Tilt' in eventName)| (i==1)):
            WeightsMatrix[:, eventIndices] = 1/ P[dt]['nObsPoints']
        else:
            eventFilter = FullParamHis['EventNum'].values==i
            #x
            Xidx = (eventFilter) & (FullParamHis['Prop']=='X')
            xmin = FullParamHis['minV'].loc[Xidx].values[-1]
            xmax = FullParamHis['maxV'].loc[Xidx].values[-1]

            #y
            Yidx = (eventFilter) & (FullParamHis['Prop']=='Y')
            ymin = FullParamHis['minV'].loc[Yidx].values[-1]
            ymax = FullParamHis['maxV'].loc[Yidx].values[-1]

            #yaxis
            yaxisIdx = (eventFilter) & (FullParamHis['Prop']=='YAxis')
            a = FullParamHis['maxV'].loc[yaxisIdx].values[-1]

            zaxisIdx = (eventFilter) & (FullParamHis['Prop']=='ZAxis')
            b = FullParamHis['maxV'].loc[zaxisIdx].values[-1]

            dipdirectionIdx = (eventFilter) & (FullParamHis['Prop']=='Dip Direction')
            alpha = FullParamHis['maxV'].loc[dipdirectionIdx].values[-1]
            alpha = np.deg2rad(-(alpha+90))

            if('Fault' in eventName):
               
                nTimes = int(np.max([(xmax-xmin)/b,10]))
                Xs = np.linspace(xmin,xmax,nTimes)
                Ys = np.linspace(ymin,ymax,nTimes)
                for x in range(len(Xs)):
                    for y in range(len(Ys)):                        
                        tX = xObs-Xs[x]
                        tY = yObs-Ys[y]
                        condition1 = (np.power(tX*np.cos(alpha)+tY*np.sin(alpha), 2)/a**2+np.power(tX*np.sin(alpha)-tY*np.cos(alpha),2)/b**2) < 1
                        WeightsMatrix[np.ix_(condition1, eventIndices)] = 1
            # find the local error for the intrusion
            else:
                R = np.max([a, b])

                nTimes = int(np.max([(xmax-xmin)/b,10]))
                Xs = np.linspace(xmin,xmax,nTimes)
                Ys = np.linspace(ymin,ymax,nTimes)
                for x in range(len(Xs)):
                    for y in range(len(Ys)):                        
                        tX = xObs-Xs[x]
                        tY = yObs-Ys[y]
                        condition1 = (np.power(tX, 2)+np.power(tY,2)) < (R**2)
                        WeightsMatrix[np.ix_(condition1, eventIndices)] = 1

    #sometimes there can be errors in the data. To avoid that problem, we give those guys zeros weight
    WeightsMatrix[np.isnan(WeightsMatrix)] = 0
    sumWeightsPerParam = np.sum(WeightsMatrix, axis=0)
    filterSumZero = sumWeightsPerParam==0
    WeightsMatrix[:, ~filterSumZero] = WeightsMatrix[:, ~filterSumZero]/sumWeightsPerParam[~filterSumZero]
    WeightsMatrix[:, filterSumZero] = 1/ P[dt]['nObsPoints']

    return WeightsMatrix

##########################################
## Output functions
##########################################

def OutputImageAndHisFile(P, figfilename='', hisfilename=''):
 
    if(figfilename==''):
        Error=get_combo_err(P, -1)
        figfilename = P['folder']+'VisualInspection/Viz_T'+str(P['HypP']['thread_num'])+'_G_' + str(P['iterationNum']) +'_Err_'+ '{:.0f}'.format(Error*1000)+'.png'
    
    Viz.visualize_opt_step(figfilename, P)

    if(hisfilename==''):
        Error=get_combo_err(P, -1)
        hisfilename = P['folder']+'HistoryFileInspection/His_'+str(P['HypP']['thread_num'])+'_G_' + str(P['iterationNum'])+'_Err_'+'{:.0f}'.format(Error*1000)+'.his'
    
    copyfile(P['SampledInputFileName'], hisfilename) 


def SaveResults2Files(P):

    if(P['HypP']['ErrorType']=='Local'):
        np.savetxt(P['folder']+'ErrorPerParameter_His', get_combo_param_err_matrix(P), delimiter=',')        
    
    DictMismatch = {}
    for dt in P['DataTypes']:
        DictMismatch[dt+'_Error'] = P[dt]['L1MismatchList']

    MismatchListPD = pd.DataFrame(DictMismatch)
    TotalError = get_combo_err_list(P)
    MismatchListPD['Mismatch']= TotalError
 
    AllAcceptedListPD = pd.DataFrame({'AllAcceptedList': P['AllAcceptedList']})
    MismatchListPD.to_csv(P['folder']+'Mismatch.csv', index=False)

    vNames=(P['ModelParamTable']['EventName']+'_'+P['ModelParamTable']['Prop']).values
    iterNums = np.char.mod('%d', np.arange(P['iterationNum']+1))
    parametersVals = P['ModelParamTable'][iterNums].values.T
    ParameterHis2File = pd.DataFrame(parametersVals, columns = vNames)

    vNames=(P['FullParamHis']['EventName']+'_'+P['FullParamHis']['Prop']).values
    iterNums = np.char.mod('%d', np.arange(P['iterationNum']+1))
    parametersVals = P['FullParamHis'][iterNums].values.T
    FullParameterHis2File = pd.DataFrame(parametersVals, columns = vNames)

    for dt in P['DataTypes']:
        ParameterHis2File[dt+'_Error']= P[dt]['L1MismatchList']
        FullParameterHis2File[dt+'_Error']= P[dt]['L1MismatchList']
    
    ParameterHis2File['TotalError']= TotalError
    FullParameterHis2File['TotalError']= TotalError
        
    ParameterHis2File.to_csv(P['folder']+'ParameterHis.csv', index=False)
    FullParameterHis2File.to_csv(P['folder']+'FullParameterHis.csv', index=False)
    AllAcceptedListPD.to_csv(P['folder']+'AllAcceptedList.csv', index=False)

    DictNormalizePD = pd.DataFrame(P['DatNormCoef'], index=[0])
    DictNormalizePD.to_csv(P['folder']+'NormFactor.csv')
    
    for p in Path(P['folder']).glob("noddy_out*"):
        p.unlink()

def getfaultXY(ModelParamTable, eventNum,P, colName):

    faultProp = ['ZAxis', 'X', 'Y', 'Z', 'Dip Direction', 'Dip', 'Amplitude']
    idxF = ModelParamTable['EventNum']==eventNum
    faultP = {}

    for prop in ['PtX','PtY']:
        Idx = idxF & (ModelParamTable['Prop'].str.contains(prop))
        faultP[prop] = ModelParamTable.loc[Idx, colName].values
    
    for prop in faultProp:
        Idx = idxF & (ModelParamTable['Prop']==prop)
        faultP[prop] = ModelParamTable.loc[Idx, colName].values[0]

            
    x= (faultP['PtX']/628)*faultP['ZAxis']*2-faultP['ZAxis']    
    y= (faultP['PtY']/100)*faultP['Amplitude']

    x = x+faultP['X']
    y = y+faultP['Y']
    
    pts = np.hstack((np.asarray(x).reshape(-1,1), np.asarray(y).reshape(-1,1)))
    rotatedPts = rotate(pts, origin=np.asarray([faultP['X'], faultP['Y']]).reshape(1,2), degrees=-faultP['Dip Direction'])
    x = rotatedPts[:,0]
    y = rotatedPts[:,1]

    zSlice = 4000
    
    diffZ = zSlice - faultP['Z']


    if((faultP['Dip Direction']>0)&(faultP['Dip']>=90)):
        faultP['Dip'] = 180-faultP['Dip']
        faultP['Dip Direction'] = np.mod(faultP['Dip Direction']+180,360)
    elif((faultP['Dip Direction']<0)&(faultP['Dip']<=90)):
        faultP['Dip Direction'] = faultP['Dip Direction']+360
    elif((faultP['Dip Direction']<0)&(faultP['Dip']>90)):
        faultP['Dip'] = 180-faultP['Dip']
        faultP['Dip Direction'] = np.mod(faultP['Dip Direction']+180,360)           
    
    hypotenuse = 1./np.tan(np.deg2rad(faultP['Dip']))
    if(faultP['Dip Direction']<=90):
        dispVecX =  -np.cos(np.deg2rad(90-faultP['Dip Direction']))*hypotenuse
        dispVecY =  -np.sin(np.deg2rad(90-faultP['Dip Direction']))*hypotenuse
    elif(faultP['Dip Direction']<=180):
        dispVecX =  -np.sin(np.deg2rad(180-faultP['Dip Direction']))*hypotenuse
        dispVecY =  np.cos(np.deg2rad(180-faultP['Dip Direction']))*hypotenuse
    elif(faultP['Dip Direction']<=270):
        dispVecX =  np.cos(np.deg2rad(270-faultP['Dip Direction']))*hypotenuse
        dispVecY =  np.sin(np.deg2rad(270-faultP['Dip Direction']))*hypotenuse
    else: #rotation<360
        dispVecX =  np.sin(np.deg2rad(360-faultP['Dip Direction']))*hypotenuse
        dispVecY =  -np.cos(np.deg2rad(360-faultP['Dip Direction']))*hypotenuse
    
    x=x+dispVecX*diffZ
    y=y+dispVecY*diffZ

    #account for origin
    x = x+P['xmin']
    y = y+ P['ymin']

    xy = np.concatenate((x.reshape(-1,1),y.reshape(-1,1)), axis=1)

    return xy, dispVecX, dispVecY   

def rotate(p, origin=(0, 0), degrees=0):
    # x and y are columns
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return np.squeeze((R @ (p.T-origin.T) + origin.T).T)


def GetFaultsXY(P):
    nFaults = P['nFaults']
    xList=[]
    yList=[]
    fault_ballsX=[]
    fault_ballsY=[]
    
    for i in range(nFaults):
        faultN = i+P['nNonFaultingEvents']+1
        colName = str(P['iterationNum'])
        xy, dispVecX, dispVecY = getfaultXY(P['FullParamHis'], faultN, P, colName)
        x = xy[:,0]
        y = xy[:,1]
        xList.append(x)
        yList.append(y)

        if(np.mod(len(x),2)==1):
            midpointIdx = int(np.floor(len(x)/2))
            midX = x[midpointIdx]
            midY = y[midpointIdx]
        else:
            idx1 = int(np.floor((len(x)-1)/2))
            x1 = x[idx1]
            x2=x[idx1+1]
            midX = (x1+x2)/2
            
            y1 = y[idx1]
            y2 = y[idx1+1]
            midY = (y1+y2)/2

        distMoveBall=120
        Multiplier = np.sqrt(distMoveBall**2/(dispVecX**2+dispVecY**2))
        midX=midX-dispVecX*Multiplier        
        midY=midY-dispVecY*Multiplier        
        
        fault_ballsX.append(midX)
        fault_ballsY.append(midY)


    FaultsXY={'x': xList, 'y':yList, 'fault_ballsX':fault_ballsX, 
              'fault_ballsY':fault_ballsY}            

    return FaultsXY


##########################################
## Error access functions
##########################################

## Combined global errors (for optimization and visualization (L1))
def get_combo_err(P, idx, errNorm = 'L1', datawts=None):
    '''Normalize, weight and combine the error of different data types'''
    
    combined_error = 0
    for i in range(len(P['DataTypes'])):
        datatype = P['DataTypes'][i]
        errpoints = np.asarray(P[datatype]['L1MismatchMatrix'][:, idx])
        errpointsnorm = errpoints/P['DatNormCoef'][datatype]
        if(errNorm=='L2'):
            errpointsnorm = errpointsnorm**2
        elif(errNorm=='Lhalf'):
            errpointsnorm = errpointsnorm**0.5
        err = np.mean(errpointsnorm)
        if(datawts==None):
            datawt = 1./len(P['DataTypes'])
        else: 
            datawt = datawts[datatype]
        err_weighted = err*datawt
        combined_error = combined_error+err_weighted
    
    return combined_error

def get_combo_err_list(P, errNorm = 'L1', datawts=None):
    '''Normalize, weight and combine the error of different data types'''
    
    combined_error = 0
    for i in range(len(P['DataTypes'])):
        datatype = P['DataTypes'][i]
        errpoints = np.asarray(P[datatype]['L1MismatchMatrix'])
        errpointsnorm = errpoints/P['DatNormCoef'][datatype]
        if(errNorm=='L2'):
            errpointsnorm = errpointsnorm**2
        elif(errNorm=='Lhalf'):
            errpointsnorm = errpointsnorm**0.5
        err = np.mean(errpointsnorm, axis=0)
        if(datawts==None):
            datawt = 1./len(P['DataTypes'])
        else: 
            datawt = datawts[datatype]
        err_weighted = err*datawt
        combined_error = combined_error+err_weighted
    
    return combined_error

def get_combo_param_err_indices(P, indices, errNorm = 'L1', datawts=None):
    '''calculate the combined local error at specified step indices'''
    
    nParam = P['nParam']          
    ErrorPerParameter = np.zeros((nParam,1))
    for i in range(nParam):
        idx = indices[i]
        LastMismatchReCalibrated = 0
        for dt in P['DataTypes']:
            errpoints = P[dt]['L1MismatchMatrix'][:, idx]
            errpointsnorm = errpoints/P['DatNormCoef'][dt]
            if(errNorm=='L2'):
                errpointsnorm = errpointsnorm**2
            elif(errNorm=='Lhalf'):
                errpointsnorm = errpointsnorm**0.5           
            local_err = errpointsnorm*P[dt]['LocalErrorWeights'][:, i]
            if(datawts==None):
                datawt = 1./len(P['DataTypes'])
            else: 
                datawt = datawts[dt]
            weighted_err = local_err*datawt
            LastMismatchReCalibrated = LastMismatchReCalibrated + weighted_err
        ErrorPerParameter[i] = LastMismatchReCalibrated

    return ErrorPerParameter  

def get_combo_param_err_idx(P, idx, errNorm='L1', datawts=None):
    '''calculate the combined local error'''
            
    nParam = P['nParam']
    ErrorPerParameter = np.zeros((nParam,P['iterationNum']))
    for dt in P['DataTypes']:
        err = P[dt]['L1MismatchMatrix'][:, idx]
        norm_err = err/P['DatNormCoef'][dt]
        if(errNorm=='L2'):
            norm_err = norm_err**2
        elif(errNorm=='Lhalf'):
            norm_err = norm_err**0.5           
        local_err = norm_err*P[dt]['LocalErrorWeights']
        if(datawts==None):
            datawt = 1./len(P['DataTypes'])
        else: 
            datawt = datawts[dt]
        weighted_err = local_err*datawt
        ErrorPerParameter = ErrorPerParameter + weighted_err

    return ErrorPerParameter

def get_combo_param_err_matrix(P, errNorm='L1'):
    '''calculate the combined local error'''
    
    nParam = P['nParam']
    ErrorPerParameter = np.zeros((nParam,P['iterationNum']+1))
    for dt in P['DataTypes']:
        err = P[dt]['L1MismatchMatrix']
        norm_err = err/P['DatNormCoef'][dt]
        if(errNorm=='L2'):
            norm_err = norm_err**2
        elif(errNorm=='Lhalf'):
            norm_err = norm_err**0.5           
        local_err = np.dot(P[dt]['LocalErrorWeights'].T, norm_err)
        weighted_err = local_err*P['dat_opt_wts'][dt]
        ErrorPerParameter = ErrorPerParameter + weighted_err

    return ErrorPerParameter

##########################################
## Check if terminate
##########################################
def CheckEarlyStop(P):
    '''check whether the MCMC loop has converged and stop if converged'''

    idx=P['iterationNum']
    MismatchList = get_combo_err_list(P)
    BreakEarly=0

    if(idx>100):
        meanLastRun = np.mean(MismatchList[idx-100:idx-50])
        meanThisRun = np.mean(MismatchList[idx-50:idx])
        PercentImproveMean = (meanLastRun - meanThisRun)/meanLastRun     

        minLastRun = np.min(MismatchList[idx-100:idx-50])
        minThisRun = np.min(MismatchList[idx-50:idx])
        PercentImproveMin = (minLastRun - minThisRun)/minLastRun     

        if((PercentImproveMin<0.02)&(PercentImproveMean<0.02)):
            BreakEarly=1
            
    return BreakEarly
