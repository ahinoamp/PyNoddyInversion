# -*- coding: utf-8 -*-
"""
Created September 17, 2020

@author: ahinoamp@gmail.com

This script provides functions for sampling parameters values that are used to
create history files for running the kinematic simulator Noddy.
"""
import numpy as np
import pandas as pd
import random
import re

import HisTemplates as his
import GeneralInversionUtil as GI

def CreatePyNoddyTemplateGivenParameterTablePatua(P,ModelParamTable):
    '''Create a template of a Noddy history file, where parameters that are 
    changing are marked as follows $parameter$ and these files can easily be 
    updated after sampling new parameters'''
    
    nEvents = len(np.unique(ModelParamTable['EventNum']))
    nFaults = nEvents-5
    FaultPoints = P['nFaultPoints']
    # This line needs to be changed when considering scenarios different than 
    # the one presented in this work
    faultingEvents = ModelParamTable['EventNum']>5
    faultnames = pd.unique(ModelParamTable.loc[faultingEvents, 'EventName'])
    P['FaultNames']=faultnames
    
    filename = P['SampledInputFileName'] + 'template'    
    CreatePyNoddyTemplate(filename, P, nLayers = 4, nFaults = nFaults, faultnames = faultnames,
                          FaultPoints = FaultPoints, 
                          LayerNames = ['Intrusive', 'Felsic', 'Mafic', 'Sed'], 
                          origin=[0,0,4000], extent=P['HypP']['xy_extent'])
    
    # Find the non changing parameters (with low standard deviation)
    # and replace those files in the template
    nonChangingParametersF = ModelParamTable['std']<0.00001
    nonChangingParameters = ModelParamTable[nonChangingParametersF]
    
    P['NonChangingProps'] = nonChangingParameters
    
    Name = '$' + nonChangingParameters['EventName']+'_'+ nonChangingParameters['Prop']+'$'
    Values = nonChangingParameters['minV'].apply(str)
    substitutions = dict(zip(Name, Values)) 
    if(len(substitutions)==0):
        pass
    else:
        ReplaceValuesInTemplate(filename=filename, substitutions=substitutions)
        
def ProposeParameters(P):
    ''' 
    Sample parameters as part of the optimisation workflow, and generate a 
    history file using the sampled parameters. The parameters are samples by
    taking a proposal step from the last accepted parameters.
    '''

    # sample parameters
    # If this is the first round, then the values are sampled
    # between the minimum and maximum values for the parameter    
    if(P['iterationNum']==0):
        parameters = SampleParametersMinMax(P)
    else:
        parameters = SampleParameters(P['ModelParamTable'], P, P['iterationNum'])
    
    # write paramters to file
    UpdateTemplateGivenParam(parameters, P['ModelParamTable'],
                             P['SampledInputFileName'], P)
    
    
    # book keep
    P['ModelParamTable'][str(P['iterationNum'])] = parameters
    P['FullParamHis'].loc[P['OptimizeParametersIdx'],str(P['iterationNum'])] = parameters
    P['FullParamHis'].loc[~P['OptimizeParametersIdx'],str(P['iterationNum'])] = P['FullParamHis'].loc[~P['OptimizeParametersIdx'],'minV']
    
    return parameters


def SampleParametersMinMax(P):
    ''' Sample parameters by selecting values randomly between the minimum and
    maximum values'''
    ModelParamTable = P['ModelParamTable']
    
    nParam2Set = len(ModelParamTable)
    Properties = ModelParamTable['Prop']
    MinV = ModelParamTable['minV']
    MaxV = ModelParamTable['maxV']
    DistTypes = ModelParamTable['Dist']
    EventN = ModelParamTable['EventNum']

    paraValList = []
    
    # Loop through all the parameters and set them
    i=0
    while i <(nParam2Set):
        prop = Properties[i]
        minV = MinV[i]
        maxV = MaxV[i]
        DistType= DistTypes[i]
        # If the property being sampled is the slip, make sure the slip isn't
        # larger than 0.2*faultlength
        if('Slip' in prop):
            maxV = np.min([maxV, 2000, 0.35*XAxis])
        # If the property is magnetic susceptability, the numbers are drawn 
        # from the log of the values range
        if(DistType=='LogGaussian'):
            minVlog = np.log10(minV)
            maxVlog = np.log10(maxV)
            randomNumber = np.random.rand()
            paramVallog = randomNumber*(maxVlog-minVlog)+minVlog
            paramVal = np.power(10, paramVallog)  
        else:
            randomNumber = np.random.rand()
            paramVal = randomNumber*(maxV-minV)+minV
        paraValList.append(paramVal)
        if('XAxis' in prop):
            XAxis = paramVal
        i=i+1
    
    return paraValList

def SampleParameters(ModelParamTable, P, iterationNumber):
    '''Sample parameters by taking a step randomly around the current values'''

    nParam2Set = len(ModelParamTable)
    Properties = ModelParamTable['Prop']
    MinV = ModelParamTable['minV']
    MaxV = ModelParamTable['maxV']
    StdV = ModelParamTable['std']
    ProposalDistType = ModelParamTable['Dist']

    # Get the current parameter values
    if(P['HypP']['ErrorType']=='Global'):
        CurrV = ModelParamTable[str(P['lastAcceptedIdx'])]
    else:
        CurrV = np.zeros((P['nParam'],))
        for i in range(P['nParam']):
            CurrV[i] = ModelParamTable.loc[i, str(P['lastAcceptedIdx'][i])]

    paraValList = []
    
    # Loop through the parameters and sample a new value around the current 
    # values
    i=0
    while i <(nParam2Set):
        prop = Properties[i]
        currV = CurrV[i]
        minV = MinV[i]
        maxV = MaxV[i]
        stdV = StdV[i]

        proposalDistType = ProposalDistType[i]
        
        # Make sure the slip is at most 0.2*faultlength 
        if('Slip' in prop):
            maxV = np.min([maxV, 2000, 0.2*XAxis])

        # Exploration rate decreases and increases the standard deviation
        # based on the error size
        ExplorationRate = get_exploration_rate(P, i)
        stdV = stdV*ExplorationRate

        paramVal = MakeProposal(currV, proposalDistType, stdV, minV, maxV)
        paraValList.append(paramVal)
        if('XAxis' in prop):
            XAxis = paramVal            
        i=i+1
    
    return paraValList

def MakeProposal(InitialValue, ProposalDistType, STD_Range, minV, maxV):

    if(ProposalDistType=='uniform'):
        randomNumber = np.random.rand()
        NewValue = InitialValue - STD_Range + randomNumber*2*STD_Range
    elif(ProposalDistType=='LogGaussian'):
        minVlog = np.log10(minV)
        maxVlog = np.log10(maxV)
        Steplog =  (maxVlog-minVlog)*STD_Range   
        currentVallog = np.log10(InitialValue)
        NewValuelog = np.random.normal(currentVallog, Steplog, 1)[0]
        NewValue = np.power(10, NewValuelog)   
    else: #gaussian
        NewValue = np.random.normal(InitialValue, STD_Range, 1)[0]

    if NewValue<minV:
        NewValue=minV
        
    if NewValue>maxV:
        NewValue = maxV
        
    return NewValue


def get_exploration_rate(P, i):
    '''The exploration rate determines how much to increase the step size when
    performing an optimisation based on how much the error has decreased'''

    if((P['HypP']['ExplorationRate']=='None')| (P['iterationNum']==0)):
        ExplorationRate=1
    #LinearErrorBased exploration
    elif(P['HypP']['ErrorType']=='Global'):
        last_err = GI.get_combo_err(P, idx=-1,
                                           errNorm = P['HypP']['ErrorNorm'],
                                           datawts = P['data_wts'])
        ExplorationRate = np.min([1.2, P['HypP']['SteppingSizeMult']*last_err])
    #Error type is local
    else:
        lastErrorParameter = GI.get_combo_param_err_idx(P, idx=-1, 
                                           errNorm = P['HypP']['ErrorNorm'],
                                           datawts = P['data_wts'])
        
        ExplorationRate = np.min([1.2, np.mean(P['HypP']['SteppingSizeMult']*lastErrorParameter)])
    
    return ExplorationRate

    
def UpdateTemplateGivenParam(paraValList, ModelParamTable, 
                                  OutputfileName, P):
    '''replace values in the history file template given parameters values'''

    Name = '$' + ModelParamTable['EventName']+'_'+ ModelParamTable['Prop']+'$'
    paraValList = ["{:.3e}".format(i) for i in paraValList]
    substitutions = dict(zip(Name, paraValList)) 

    # the layer thicknesses need to be transformed to height
    sedThickness = float(substitutions['$Sed_Thickness$'])
    maficThickness = float(substitutions['$Mafic_Thickness$'])
    felsicThickness = float(substitutions['$Felsic_Thickness$'])
    ModelTop = P['HypP']['xy_extent'][2] 
    substitutions['$Sed_Height$'] = "{:.5f}".format(ModelTop-sedThickness)
    substitutions['$Mafic_Height$']=  "{:.5f}".format(ModelTop-sedThickness-maficThickness)
    substitutions['$Felsic_Height$']=  "{:.5f}".format(ModelTop-sedThickness-maficThickness-felsicThickness)
    substitutions['$Intrusive_Height$']= str(0)

    # the zaxis needs to be the same as xaxis
    # this is a funny bug in Noddy that has to be fixed
    # when you use elliptic faults, the shape is determined by the x direction
    # butthe actual x direction is the z direction...
    nFaults = np.sum(ModelParamTable['EventName'].drop_duplicates().str.contains('Fault'))
    listV = substitutions.keys()
    for i in range(nFaults):
        if((('$Fault'+str(i)+'_ZAxis$') in listV) & (('$Fault'+str(i)+'_XAxis$') in listV)): 
            substitutions['$Fault'+str(i)+'_ZAxis$']= substitutions['$Fault'+str(i)+'_XAxis$']

    file1 = open(OutputfileName + 'template',"r") 
    string = file1.read()
    output = replace(string, substitutions)
    file1.close
    
    file1 = open(OutputfileName,"w") 
    file1.write(output)
    file1.close    

def ReplaceValuesInTemplate(filename, substitutions):
    file1 = open(filename,"r") 
    string = file1.read()
    output = replace(string, substitutions)
    file1.close
    
    file1 = open(filename,"w") 
    file1.write(output)
    file1.close
    
def replace(string, substitutions):

    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

def CreatePyNoddyTemplate(filename, P, nLayers = 4, nFaults = 5, faultnames =['f0', 'f1', 'f2', 'f3', 'f4', 'f5'],
                          FaultPoints = [3,5,6,23,19],
                          LayerNames = ['Intrusive', 'Felsic', 'Mafic', 'Sed'], 
                          origin = [0,0,4000], extent=[9000,9000,4000]):
    nEvents = nFaults + 5
    
    FaultProperties = ['X', 'Y', 'Z', 'Dip Direction', 
                       'Dip', 'Slip', 'Rotation', 'Amplitude', 'Radius', 'XAxis', 'YAxis',
                       'ZAxis','Profile Pitch', 'Pitch'] 
    
    file1 = open(filename,"w") 
    headerTxt = his._HisTemplates().header
    file1.write(headerTxt+ '\n') 
    
    numEventsText = "No of Events\t= %d\n" % (nEvents)
    file1.write(numEventsText) 

    EventTitle = 'Event #1	= STRATIGRAPHY'
    file1.write(EventTitle + '\n') 
    SubTitle = "	Num Layers = %d" % (nLayers) 
    file1.write(SubTitle + '\n') 
    for i in range(nLayers):
        stratTxt = his._HisTemplates().strati_layer
        stratTxt = stratTxt.replace("$NAME$", LayerNames[i])
        stratTxt = stratTxt.replace("$RED$", str(random.randrange(0, 255)))
        stratTxt = stratTxt.replace("$GREEN$", str(random.randrange(0, 255)))
        stratTxt = stratTxt.replace("$BLUE$", str(random.randrange(0, 255)))
        stratTxt = stratTxt.replace("$Height$", '$' + LayerNames[i] +'_Height$')
        stratTxt = stratTxt.replace("$Density$", '$' + LayerNames[i] +'_Density$')
        stratTxt = stratTxt.replace("$MagSus$", '$' + LayerNames[i] +'_MagSus$')        
        file1.write(stratTxt + '\n') 
    file1.write("	Name	= Strat\n")

    EventTitle = 'Event #2	= TILT'    
    file1.write(EventTitle + '\n') 
    tiltTxt = his._HisTemplates().tilt
    file1.write(tiltTxt+ '\n') 

    EventTitle = 'Event #3	= PLUG'    
    file1.write(EventTitle + '\n') 
    plugTxt = his._HisTemplates().plug
    plugTxt = plugTxt.replace('$Plug_X$', '$Plug'+str(0)+'_X$')
    plugTxt = plugTxt.replace('$Plug_Y$', '$Plug'+str(0)+'_Y$')
    plugTxt = plugTxt.replace('$Plug_Z$', '$Plug'+str(0)+'_Z$')
    plugTxt = plugTxt.replace('$Plug_XAxis$', '$Plug'+str(0)+'_XAxis$')
    plugTxt = plugTxt.replace('$Plug_YAxis$', '$Plug'+str(0)+'_YAxis$')
    plugTxt = plugTxt.replace('$Plug_ZAxis$', '$Plug'+str(0)+'_ZAxis$')
    plugTxt = plugTxt.replace('$Plug_Density$', '$Plug'+str(0)+'_Density$')
    plugTxt = plugTxt.replace('$Plug_MagSus$', '$Plug'+str(0)+'_MagSus$')
    plugTxt = plugTxt.replace('$Plug_Radius$', '$Plug'+str(0)+'_Radius$')
    plugTxt = plugTxt.replace('$Plug_Dip Direction$', '$Plug'+str(0)+'_Dip Direction$')
    file1.write(plugTxt+ '\n') 
    
    EventTitle = 'Event #4	= PLUG'    
    file1.write(EventTitle + '\n') 
    plugTxt = his._HisTemplates().plug
    plugTxt = plugTxt.replace('$Plug_X$', '$Plug'+str(1)+'_X$')
    plugTxt = plugTxt.replace('$Plug_Y$', '$Plug'+str(1)+'_Y$')
    plugTxt = plugTxt.replace('$Plug_Z$', '$Plug'+str(1)+'_Z$')
    plugTxt = plugTxt.replace('$Plug_XAxis$', '$Plug'+str(1)+'_XAxis$')
    plugTxt = plugTxt.replace('$Plug_YAxis$', '$Plug'+str(1)+'_YAxis$')
    plugTxt = plugTxt.replace('$Plug_ZAxis$', '$Plug'+str(1)+'_ZAxis$')
    plugTxt = plugTxt.replace('$Plug_Density$', '$Plug'+str(1)+'_Density$')
    plugTxt = plugTxt.replace('$Plug_Radius$', '$Plug'+str(1)+'_Radius$')
    plugTxt = plugTxt.replace('$Plug_MagSus$', '$Plug'+str(1)+'_MagSus$')
    plugTxt = plugTxt.replace('$Plug_Dip Direction$', '$Plug'+str(1)+'_Dip Direction$')
    file1.write(plugTxt+ '\n') 
    
    EventTitle = 'Event #5	= PLUG'    
    file1.write(EventTitle + '\n') 
    plugTxt = his._HisTemplates().plug
    plugTxt = plugTxt.replace('$Plug_X$', '$Plug'+str(2)+'_X$')
    plugTxt = plugTxt.replace('$Plug_Y$', '$Plug'+str(2)+'_Y$')
    plugTxt = plugTxt.replace('$Plug_Z$', '$Plug'+str(2)+'_Z$')
    plugTxt = plugTxt.replace('$Plug_XAxis$', '$Plug'+str(2)+'_XAxis$')
    plugTxt = plugTxt.replace('$Plug_YAxis$', '$Plug'+str(2)+'_YAxis$')
    plugTxt = plugTxt.replace('$Plug_ZAxis$', '$Plug'+str(2)+'_ZAxis$')
    plugTxt = plugTxt.replace('$Plug_Density$', '$Plug'+str(2)+'_Density$')
    plugTxt = plugTxt.replace('$Plug_Radius$', '$Plug'+str(2)+'_Radius$')
    plugTxt = plugTxt.replace('$Plug_MagSus$', '$Plug'+str(2)+'_MagSus$')
    plugTxt = plugTxt.replace('$Plug_Dip Direction$', '$Plug'+str(2)+'_Dip Direction$')
    file1.write(plugTxt+ '\n') 
    
    for i in range(nFaults):
        FaultEventName = faultnames[i]
        nPoints= FaultPoints[i]
        EventTitle = 'Event #%d	= FAULT' % (i+6)  
        file1.write(EventTitle + '\n') 

        #start
        faultTxt = his._HisTemplates().fault_start
        for prop in FaultProperties:
            faultTxt = faultTxt.replace("$"+prop+"$", '$'+FaultEventName+'_'+prop+'$')
        faultTxt = faultTxt.replace('$Join Type$', P['JoinType'])
        file1.write(faultTxt+ '\n') 

        #middle            
        faultPointTxt= "    Num Points    = %d" % (nPoints) 
        file1.write(faultPointTxt+ '\n')
        for p in range(nPoints):
            ptX = " 		Point X = $"+FaultEventName+"_PtX"+str(p)+'$'
            file1.write(ptX+ '\n') 
            ptY = " 		Point Y = $"+FaultEventName+"_PtY"+str(p)+'$'
            file1.write(ptY+ '\n')             

        #end            
        faultTxt = his._HisTemplates().fault_end
        faultTxt = faultTxt.replace("$NAME$", FaultEventName)
        file1.write(faultTxt+ '\n') 
            
    footerTxt = his._HisTemplates().footer
    footerTxt = footerTxt.replace('$origin_z$', str(origin[2]))
    footerTxt = footerTxt.replace('$extent_x$', str(extent[0]))
    footerTxt = footerTxt.replace('$extent_y$', str(extent[1]))
    footerTxt = footerTxt.replace('$extent_z$', str(extent[2]))
    footerTxt = footerTxt.replace('$cube_size$', str(P['cubesize']))
    file1.write(footerTxt) 
    
    file1.close()