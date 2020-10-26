# -*- coding: utf-8 -*-
"""
Created September 15, 2020

@author: ahinoamp@gmail.com

This script sets up the model paramters table
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import interpolate
import random   
from shapely.geometry import LineString
import GeneralInversionUtil as GI

def DefinePriorUncertainty(P):
    '''This method creates the ModelParamTable, that defines the different
    properties asscoiated with each event and the minimum and maximum ranges of
    values for those properties, as well as the stepping size when sampling
    a new value for that parameter'''

    # Create lists to fill the rows of model parameter table with different
    # attributes of the parameters
    PropAttr = {}
    PropAttr['Scenario'] = []  
    PropAttr['EventNum'] = []
    PropAttr['EventName'] = []
    PropAttr['Prop'] = []
    PropAttr['Dist'] =[]
    PropAttr['minV'] = []
    PropAttr['maxV'] = []
    PropAttr['std'] = []
    P['PropAttr'] = PropAttr
    
   # Add stratigraphy event
    add_strat_event(P) 
    add_tilt_event(P)
    add_plug_events(P)
    add_faulting_events(P)

    ModelParamTable = pd.DataFrame(PropAttr)
    
    return ModelParamTable


def add2Table(P, scenario_scalar, eventN_scalar, event_name,
              prop, dist, minV, maxV, std):
    
    P['PropAttr']['Scenario'] += [scenario_scalar]*len(event_name)  
    P['PropAttr']['EventNum'] += [eventN_scalar]*len(event_name)
    P['PropAttr']['EventName'] += event_name
    P['PropAttr']['Prop'] += prop
    P['PropAttr']['Dist'] += dist
    P['PropAttr']['minV'] += minV
    P['PropAttr']['maxV'] += maxV
    P['PropAttr']['std'] += std

def add_strat_event(P):
    # Event 1: stratigraphy
    LayerNames = ['Sed', 'Mafic','Felsic',  'Intrusive']
    LayerDensityMin = [ 2.1, 2.2, 2.3, 2.5]
    LayerDensityMax = [ 2.36, 2.42, 2.65,  2.72]
    LayerDensityStd = [ 0.02, 0.02, 0.02,   0.02]
    nL = len(LayerNames)
    add2Table(P, 0, 1, LayerNames, ['Density']*nL, ['Gaussian']*nL, 
              LayerDensityMin, LayerDensityMax, LayerDensityStd)    
    
    LayerMagSusMin = [0.000001, 0.0001, 0.000001, 0.0001]
    LayerMagSusMax = [0.001, 0.02, 0.003,  0.03162]
    LayerMagSusStd = [0.07, 0.07, 0.07, 0.07]
    add2Table(P, 0, 1, LayerNames, ['MagSus']*nL, ['LogGaussian']*nL, 
              LayerMagSusMin, LayerMagSusMax, LayerMagSusStd)    
    
    LayerThicknessMin = [150, 250, 250]
    LayerThicknessMax = [200, 800, 800]
    LayerThickStd = [40, 50, 50]
    add2Table(P, 0, 1, LayerNames[:-1], ['Thickness']*(nL-1), ['Gaussian']*(nL-1), 
              LayerThicknessMin, LayerThicknessMax, LayerThickStd)    

def add_tilt_event(P):

    #Tilt
    TiltProp = ['Plunge Direction', 'Rotation']
    TiltMin = [0, 0]
    TiltMax = [360, 2]
    TiltStd = [8, 0.4]

    add2Table(P, 0, 2, ['Tilt']*len(TiltProp), TiltProp, 
              ['Gaussian']*len(TiltProp), TiltMin, TiltMax, TiltStd)    
    
def add_plug_events(P):

    # 3. Plug events 
    NameP0 = 'Plug0'
    ParamP0 = ['X', 'Y', 'Z', 'Radius', 'XAxis', 'YAxis', 'ZAxis', 'Density', 'MagSus', 'Dip Direction']
    MinP0 = [6950, 3980, 500, 200, 200, 300, 2500, 2.6, 0.0001, 5]
    MaxP0 = [7950, 4980, 2500,800, 800, 1000, 4000, 2.9, 0.01, 20]
    StdP0 = [P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'],
           P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'], 
           P['HypP']['XYZ_Axes_StepStd'], 0.03, 0.07, 2]
    DistP0 = ['Gaussian', 'Gaussian','Gaussian','Gaussian', 'Gaussian',
              'Gaussian', 'Gaussian', 'Gaussian', 'LogGaussian', 'Gaussian']
    add2Table(P, 0, 3, [NameP0]*len(ParamP0), ParamP0, DistP0, 
              MinP0, MaxP0, StdP0)    
    
    NameP1 = 'Plug1'
    ParamP1 = ['X', 'Y', 'Z', 'Radius', 'XAxis', 'YAxis', 'ZAxis', 'Density', 'MagSus', 'Dip Direction']
    MinP1 = [3150, 2880, 400,  50,  150,  150,  2500, 2.6, 0.0001, 0]
    MaxP1 = [4150, 3880, 2500, 400, 600, 600, 4000, 2.9, 0.01, 10]
    StdP1 = [P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'],
           P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'], 
           P['HypP']['XYZ_Axes_StepStd'], 0.03, 0.07, 2]
    DistP1 = ['Gaussian', 'Gaussian','Gaussian','Gaussian', 'Gaussian',
              'Gaussian', 'Gaussian', 'Gaussian', 'LogGaussian', 'Gaussian']
    add2Table(P, 0, 4, [NameP1]*len(ParamP0), ParamP1,DistP1, 
              MinP1, MaxP1, StdP1)    

    NameP2 = 'Plug2'
    ParamP2 = ['X', 'Y', 'Z', 'Radius', 'XAxis', 'YAxis', 'ZAxis', 'Density', 'MagSus', 'Dip Direction']
    MinP2 = [3650, 5080, 500,  0,  150,  150, 2500, 2.6, 0.0001, 0]
    MaxP2 = [4650, 6080,2500,400, 600, 600, 4000, 2.9, 0.01, 10]
    StdP2 = [P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'],
           P['HypP']['XYZ_Axes_StepStd'],P['HypP']['XYZ_Axes_StepStd'], P['HypP']['XYZ_Axes_StepStd'], 
           P['HypP']['XYZ_Axes_StepStd'], 0.03, 0.07, 2]
    DistP2 = ['Gaussian', 'Gaussian','Gaussian','Gaussian', 'Gaussian',
              'Gaussian', 'Gaussian', 'Gaussian', 'LogGaussian', 'Gaussian']
    add2Table(P, 0, 5, [NameP2]*len(ParamP2), ParamP2, DistP2, 
              MinP2, MaxP2, StdP2)    
                  
def add_faulting_events(P):
    
    
    # Add in the fault events
    nFaultPointsList = []   

    ScenarioNum = P['HypP']['ScenarioNum']
    PtGlobal_maxMove = P['HypP']['GlobalMoveEachDir']
    P['HypP']['ControlPointMovementUnits'] = 'None'
    P['JoinType']='LINES'
    Scenario = P['HypP']['ScenarioNum']
    # Don't allow the shapes of the faults to change
    NormPtXLocal_maxMove = 0
    NormPtYLocal_maxMove = 0
    RatioStepSize = 0.15
    
    # The codes of scenarios 1-5 are saved for pre-configured fault scenarios
    # otherwise scenario number refers to the number of faults
    if(Scenario<1):
        FaultData = pd.read_csv('Data/Scenario'+str(Scenario)+'_Vertices.csv').sort_values(['id'])
    else:
        FaultData = pd.read_csv('Data/FaultBank7_19.csv').sort_values(['id'])
        P['Zone']=FaultData.groupby('id').agg(zone=pd.NamedAgg(column='Zone', aggfunc='first'))['zone'].values
        P['OrigId']=FaultData.groupby('id').agg(zone=pd.NamedAgg(column='id', aggfunc='first'))['zone'].values
        
    parameterData = SetUpFaultRepresentation(FaultData.copy(), P['HypP']['xy_origin'], P['HypP']['SlipParam'])
    nFaultPointsList = nFaultPointsList + parameterData['nFaultPoints']
      
    # Loop through the faults and set the min, max, std. dev., for each of the
    # faults' properties
    parameters = ['X', 'Y', 'Z', 'XAxis', 'YAxis', 'ZAxis', 'Dip', 
                  'Dip Direction', 'Slip', 'Amplitude', 'Pitch', 
                  'Profile Pitch']
    nFaults = parameterData['nFaults']
    for i in range(nFaults):
        zone = parameterData['Zone'][i]
        EventName ='Fault'+str(i)+'_'+zone
        EventNumber = 6+i
        for p in range(len(parameters)):
            Prop = parameters[p]
            pVal = parameterData[Prop][i]
            
            if((Prop=='X')|(Prop=='Y')|(Prop=='Z')):
                minV = pVal-PtGlobal_maxMove
                maxV = pVal+PtGlobal_maxMove    
                stdV = P['HypP']['XYZ_Axes_StepStd']
            elif((Prop=='XAxis')|(Prop=='YAxis')|(Prop=='ZAxis')):
                minV = pVal*(1-P['HypP']['AxisRatioChange'])
                maxV = pVal*(1+P['HypP']['AxisRatioChange'])
                stdV =(maxV-minV)/10.0
            elif(Prop=='Dip'):
                minV = np.max([50, pVal-P['HypP']['DipMoveEachDirection']])
                maxV = np.min([90, pVal+P['HypP']['DipMoveEachDirection']])
                stdV = P['HypP']['Dip_StepStd']
            elif(Prop=='Dip Direction'):
                minV = pVal-P['HypP']['AzimuthMoveEachDirection']
                maxV = pVal+P['HypP']['AzimuthMoveEachDirection']
                stdV = (maxV-minV)/10.0
            elif(Prop=='Slip'):
                minV = np.max([150, pVal*0.5])
                maxV = pVal*2
                stdV = P['HypP']['Slip_StepStd']
            elif(Prop=='Amplitude'):
                minV = pVal*(1-P['HypP']['AmplitudeRatioChange'])
                maxV = pVal*(1+P['HypP']['AmplitudeRatioChange'])
                stdV = (maxV-minV)/10.0
            elif((Prop=='Pitch')|(Prop=='Profile Pitch')):
                minV = pVal
                maxV = pVal
            else:
                print('Forgot something')

            if((maxV-minV)<0.5*stdV):
                stdV=0
            add2Table(P, ScenarioNum, EventNumber, [EventName], [Prop], 
                      ['Gaussian'], [minV], [maxV], [stdV])    
    
        xVals = parameterData['PtX'][i]
        yVals = parameterData['PtY'][i]
        nFaultPoints = len(xVals)
        for p in range(nFaultPoints):
            minV = xVals[p]-NormPtXLocal_maxMove
            maxV = xVals[p]+NormPtXLocal_maxMove
            stdV = (maxV-minV)*RatioStepSize
            FaultArrayName = 'PtX'+str(p)

            add2Table(P, ScenarioNum, EventNumber, [EventName], [FaultArrayName], 
                      ['Gaussian'], [minV], [maxV], [stdV])    


            minV = np.max([yVals[p]-NormPtYLocal_maxMove, -100])
            maxV = np.min([yVals[p]+NormPtYLocal_maxMove, 100])
            stdV = (maxV-minV)*RatioStepSize
            FaultArrayName = 'PtY'+str(p)

            add2Table(P, ScenarioNum, EventNumber, [EventName], [FaultArrayName], 
                      ['Gaussian'], [minV], [maxV], [stdV])    

    P['nFaultPoints'] = parameterData['nFaultPoints']        
    
    
def SetUpFaultRepresentation(Data, xy_origin, SlipParam=0.04, nPointsDivideList='null'):    
    if(type(nPointsDivideList) is str):
        nPointsDividePreDetermined=0 
    else:
        nPointsDividePreDetermined=1 

    fault_params={}
    fault_params['nFaultPoints']=[]
    fault_params['PtX'] = []
    fault_params['PtY'] = []
    fault_params['X'] = []
    fault_params['Y'] = []
    fault_params['Z'] = []
    fault_params['XAxis'] = []
    fault_params['YAxis'] = []
    fault_params['ZAxis'] = []
    fault_params['Dip'] = []
    fault_params['Dip Direction'] = []
    fault_params['Slip'] = []
    fault_params['Amplitude'] = []
    fault_params['Profile Pitch'] = []
    fault_params['Pitch'] = []
    fault_params['Zone']=[]    
        
    Data['X'] = Data['X']-xy_origin[0]
    Data['Y'] = Data['Y']-xy_origin[1]
    
    Faults = pd.unique(Data['id'])
    nFaults = len(Faults)
    fault_params['nFaults']=nFaults
    
    for i in range(nFaults):
        SlipParam = random.uniform(0.05, 0.25)
        filterV = Data['id']==Faults[i]
        xy = Data.loc[filterV, ['X', 'Y']].values
        EastWest = random.choice(['East', 'West'])
        Data.loc[filterV, ['DipDirecti']].values[0,0]
        meanX = (np.max(xy[:,0])+np.min(xy[:,0]))/2
        meanY = (np.max(xy[:,1])+np.min(xy[:,1]))/2
        zone = Data.loc[filterV, ['Zone']].values[0,0]
        
        # define a matrix
        pca = PCA(2)
        # fit on data
        pca.fit(xy)
        
        
        vectorPCA1 = pca.components_[0, :]
    
        if(pca.components_[0, 0]>0):
            pca.components_[0, :] = pca.components_[0, :]*-1
        if(pca.components_[1, 1]>0):
            pca.components_[1, :] = pca.components_[1, :]*-1
            
        xypca = pca.transform(xy)
    
        vectorPCA1 = pca.components_[0, :]
        
        vectorNorth = [0,1]
    
        if(vectorPCA1[0]<0):
            vectorPCA1= vectorPCA1*-1
    
        angle = np.math.atan2(np.linalg.det([vectorPCA1,vectorNorth]),np.dot(vectorPCA1,vectorNorth))
        angle = np.degrees(angle)
        dipdirection= angle+90
        if(dipdirection<0):
            dipdirection=dipdirection+360
            
        lengthFault = np.max(xypca[:,0])-np.min(xypca[:,0])
        
        means = pca.inverse_transform([(np.max(xypca[:,0])+np.min(xypca[:,0]))/2, (np.max(xypca[:,1])+np.min(xypca[:,1]))/2])
        meanX = means[0]
        meanY = means[1]
        targetXmin = 0
        targetXmax = 628
        targetYmin = -100
        targetYmax = 100
        newRangeX = targetXmax-targetXmin
        newRangeY = targetYmax-targetYmin
        oldRangeX = (np.max(xypca[:,0])-np.min(xypca[:,0]))
        oldRangeY = (np.max(xypca[:,1])-np.min(xypca[:,1]))
        
        xypca[:,0]= ((xypca[:,0]-np.min(xypca[:,0]))/oldRangeX)*newRangeX
        if(oldRangeY<0.0001):
            pass
        else:
            xypca[:,1]= ((xypca[:,1]-np.min(xypca[:,1]))/oldRangeY)*newRangeY+targetYmin
          
        if(EastWest=='East'):
            if(dipdirection<180):
                dip = 70
            else:
                dipdirection=dipdirection-180
                xypca[:,1]=-1*xypca[:,1]
                xypca[:,0]=-1*xypca[:,0]+newRangeX
                dip = 70
            ProfilePitch=0
            Pitch=90
        elif(EastWest=='SS'):
            if(dipdirection<180):
                dip = 80
            else:
                dipdirection=dipdirection-180
                xypca[:,1]=-1*xypca[:,1]
                dip = 80
            Pitch=180
            ProfilePitch=90
        else:
            if(dipdirection>180):
                dip = 70
            else:
                dipdirection=dipdirection+180
                xypca[:,1]=-1*xypca[:,1]
                xypca[:,0]=-1*xypca[:,0]+newRangeX
                dip = 70
            ProfilePitch=0
            Pitch=90
 
        dip = random.uniform(70,85)
        xypcapd = pd.DataFrame({'X': xypca[:,0], 'Y': xypca[:,1]})
        xypcapd = xypcapd.sort_values(['X','Y'], ascending='True')
        xypca = xypcapd.values
        
        OldX = xypca[:,0]
        OldY = xypca[:,1]
        if(nPointsDividePreDetermined):
            nPointsDivide = int(nPointsDivideList[i])
        else:
            nPointsDivide = int(np.max([np.ceil(np.min([lengthFault/350,30])), 2]))
        
        newX=np.linspace(0, 628, nPointsDivide)
        f = interpolate.interp1d(OldX, OldY, kind='linear')
        newY=f(newX)

        fault_params['PtX'].append(newX)
        fault_params['PtY'].append(newY)
        fault_params['X'].append(meanX)
        fault_params['Y'].append(meanY)
        fault_params['Z'].append(random.uniform(500,5000))
        fault_params['XAxis'].append(lengthFault/2)
        fault_params['ZAxis'].append(lengthFault/2)
        fault_params['YAxis'].append(lengthFault/2)
        fault_params['Dip Direction'].append(dipdirection)
        fault_params['Dip'].append(dip)
        fault_params['Slip'].append(lengthFault*SlipParam)
        fault_params['Amplitude'].append(oldRangeY/2)
        fault_params['Profile Pitch'].append(ProfilePitch)
        fault_params['Pitch'].append(Pitch)
        fault_params['nFaultPoints'].append(len(newX))
        fault_params['Zone'].append(zone)
       
    return fault_params 

def SelectFaultingEvents(ModelParamTable, P):
    ScenarioNum=P['HypP']['ScenarioNum']
    if(ScenarioNum<1):
        filterV = ((ModelParamTable['Scenario']==0)|(ModelParamTable['Scenario']==ScenarioNum))
        ModelParamTableF = ModelParamTable[filterV].reset_index(drop=True)    
        GroupVals = ModelParamTable.groupby(['Scenario', 'EventNum']).ngroup()+1
        GroupVals = GroupVals[filterV]
        IndexChoose=np.unique(GroupVals)
        P['nNonFaultingEvents']=int(np.max(ModelParamTable['EventNum']))-len(P['nFaultPoints'])
        ###################################
        ########ALSO Below - there is an issue of no order mixing!! of events...
        ##################################
    else:
        ZoneDictionary = {'0': 'MidEast', 
                          '1': 'MidMid', 
                          '2': 'MidWest',
                          '3': 'NorthEast',
                          '4': 'NorthMid',
                          '5': 'NorthWest', 
                          '6': 'SouthEast', 
                          '7': 'SouthMid', 
                          '8': 'SouthWest'}
        invertedZoneDictionary = dict([[v,k] for k,v in ZoneDictionary.items()])
        
        FaultZones= P['Zone'].copy()
        P['nNonFaultingEvents']=int(np.max(ModelParamTable['EventNum']))-len(FaultZones)
        equivalentEventNumbers = np.arange(P['nNonFaultingEvents']+1, len(FaultZones)+P['nNonFaultingEvents']+1)
        IndexChoose=np.arange(1, P['nNonFaultingEvents']+1, dtype=int).tolist()

        Order = np.random.permutation(9)
        
        i=0
        xyAcceptedPairs = []
        trickval = 0
        while (i < ScenarioNum):
            #Look at corner Order[i]
            Zone_i = ZoneDictionary[str(Order[np.mod(i+trickval,9)])]
            #Get the groupnumbers of people in that zone
            nFaultsZone = np.sum(FaultZones==Zone_i)
            eventNinThatZone = equivalentEventNumbers[FaultZones==Zone_i]
            if(len(eventNinThatZone)==0):
                if(trickval<9):
                    trickval = trickval+1
                else:
                    print('ohhh nooo...')
                    break
            #select a random number
            rChoice = np.random.randint(nFaultsZone)
            
            #check if that's an allowed addition!

            proposedFaultEvent = eventNinThatZone[rChoice]
            if(len(IndexChoose)>0):
                truexyproposedMin, _,_ = GI.getfaultXY(ModelParamTable, 
                                               proposedFaultEvent, P, 
                                               colName='minV')
                truexyproposedMax,_,_ = GI.getfaultXY(ModelParamTable, 
                                               proposedFaultEvent, P, 
                                               colName='minV')
                truexyproposed = (truexyproposedMin+truexyproposedMax)/2
                accepted = checkLineOverlap(truexyproposed, xyAcceptedPairs)
                if(accepted==1):
                    xyAcceptedPairs.append(truexyproposed)
                else:
                    Idx2Remove = np.nonzero(equivalentEventNumbers==proposedFaultEvent)[0]
                    FaultZones= np.delete(FaultZones, [Idx2Remove])
                    equivalentEventNumbers = np.delete(equivalentEventNumbers, [Idx2Remove])
                    continue
            else:
                xyproposed = GI.getfaultXY(ModelParamTable, proposedFaultEvent,P)
                xyAcceptedPairs.append(xyproposed)
            #get the group number and add it.
            IndexChoose.append(proposedFaultEvent)
            i=i+1
            #theoretically should now remove the chosen elements from the list!
            Idx2Remove = np.nonzero(equivalentEventNumbers==proposedFaultEvent)[0]
            FaultZones= np.delete(FaultZones, [Idx2Remove])
            equivalentEventNumbers = np.delete(equivalentEventNumbers, [Idx2Remove])
        
#        nEvents = len(np.unique(GroupVals))
#        IndexChoose = np.concatenate(([1,2], random.sample(range(3,nFaultingEvents+3), ScenarioNum)))       
        EventNumbers = ModelParamTable.groupby(['Scenario', 'EventNum']).ngroup()+1
        IndexChoose = np.asarray(IndexChoose)
        
        filterV = np.isin(EventNumbers, IndexChoose)
        ModelParamTableF=ModelParamTable[filterV].reset_index(drop=True)       
        filterFaults = ModelParamTableF['EventNum']>P['nNonFaultingEvents']
        ModelParamTableF['EventNum'] = ModelParamTableF.groupby(['Scenario', 'EventNum']).ngroup()+1

        equivalentEventNumbers = np.arange(P['nNonFaultingEvents']+1, len(P['Zone'])+P['nNonFaultingEvents']+1)
        filterV = np.isin(equivalentEventNumbers, IndexChoose)
        AllUniqueZonesChosen = P['Zone'][filterV]
        if(P['verbose']):
            if(len(np.unique(AllUniqueZonesChosen))==0):
                print('oooohhh nooo')
       
    IdxChooseHyp= np.sort(IndexChoose[P['nNonFaultingEvents']:]-P['nNonFaultingEvents']-1)
#    if(P['GeneralPerturbStrategy']=='OnlyLocalXY'):
#        P['Dip Direction'] = np.asarray(P['Dip Direction']).reshape(-1,1)[IdxChooseHyp].reshape(-1,)
#    P['ZoneSelect'] = np.asarray(P['Zone']).reshape(-1,1)[IdxChooseHyp].reshape(-1,)
    P['nFaultPoints'] = np.asarray(P['nFaultPoints']).reshape(-1,1)[IdxChooseHyp].reshape(-1,)
        
    return ModelParamTableF

def checkLineOverlap(truexyproposed, xyAcceptedPairs):
    accepted = 1
    polygonProposed = LineString(truexyproposed).buffer(175)
    for i in range(len(xyAcceptedPairs)):
        polygon_i = LineString(xyAcceptedPairs[i]).buffer(175)
        overlap1 = polygonProposed.intersection(polygon_i).area / polygonProposed.area
        overlap2 = polygonProposed.intersection(polygon_i).area / polygon_i.area
        if((overlap1>0.25)|(overlap2>0.25)):
            accepted=0
            break
        
    return accepted 

def OrderFaultingEvents(ModelParamTable, P):
    FaultAzimuthsF = ((ModelParamTable['Prop']=='Dip Direction')&
                    (ModelParamTable['EventNum']>P['nNonFaultingEvents']))
    FaultAzimuths = (np.mod(ModelParamTable.loc[FaultAzimuthsF, 'minV'].values+90, 360)
                     +np.mod(ModelParamTable.loc[FaultAzimuthsF, 'maxV'].values+90, 360))/2
    EventNumbers = ModelParamTable.loc[FaultAzimuthsF, 'EventNum'].values.tolist()
    Horizontal = []
    for i in range(len(FaultAzimuths)):
        FaultAzimuth_i =FaultAzimuths[i]
        FaultEvent_i =EventNumbers[i]
        if(((FaultAzimuth_i>45)&(FaultAzimuth_i<135))|((FaultAzimuth_i>225)&(FaultAzimuth_i<315))):
            Horizontal.append(FaultEvent_i)
            randN = np.random.random(1)
            if(randN<0.7):
                EventNumbers.remove(FaultEvent_i)
                EventNumbers.append(FaultEvent_i)

    P['nFaultPoints'] = P['nFaultPoints'][np.asarray(EventNumbers)-P['nNonFaultingEvents']-1]
    EventNumbers.insert(0, 5)
    EventNumbers.insert(0, 4)
    EventNumbers.insert(0, 3)
    EventNumbers.insert(0, 2)
    EventNumbers.insert(0, 1)
    
    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(EventNumbers,range(1, len(EventNumbers)+1)))

    # Generate a rank column that will be used to sort
    # the dataframe numerically
    ModelParamTable['EventNum'] = ModelParamTable['EventNum'].map(sorterIndex)
    ModelParamTable['OrginalPos'] = np.arange(len(ModelParamTable))
    
    ModelParamTable = ModelParamTable.sort_values(['EventNum', 'OrginalPos'])

    filterFaults = ModelParamTable['EventNum']>P['nNonFaultingEvents']
    ModelParamTable.loc[filterFaults,'EventName'] = ModelParamTable.loc[filterFaults, 'EventName']+'_'+(ModelParamTable.loc[filterFaults, 'EventNum']-P['nNonFaultingEvents']-1).apply(str)

    return ModelParamTable
           
