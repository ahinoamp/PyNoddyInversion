# -*- coding: utf-8 -*-
"""
Date: September 14, 2020
@author: ahinoamp@gmail.com

This file contains the functions necessary to simulate five different data
types often found in geothermal exploration scenarios:
    1. Gravity
    2. Magnetics
    3. Granite top marker
    4. Tracer connectivity information
    5. Fault markers in the wellbore

This uses the Noddy kinematic simulator software (terminal interface) and code
taken from PyNoddy (https://github.com/cgre-aachen/pynoddy) and altered for 
better fit for this study
"""

import numpy as np
import os
import subprocess
from scipy.interpolate import interpn
import scipy
from scipy.interpolate import RegularGridInterpolator
import itertools
from scipy import interpolate

# If running on Windows, this flag ensures the Noddy window doesn't pop open
CREATE_NO_WINDOW = 0x08000000


def calculate_model(historyfile, output_name, outputoption='ALL',
                    Windows=True):
    """
    Run the Noddy executable using an input history file and options.

    Parameters
    ----------
        historyfile (str): full path to the history file *.his
        output_name (str): output name full path, e.q.: 'sandbox/output'
        outputoption (str): choose from:
                'BLOCK' --> just the geoloy
                'GEOPHYSICS'--> geology block+gravity+magnetics
                'SURFACES'--> dxf mesh file of geology
                'BLOCK_GEOPHYS'--> geology + geophysics
                'BLOCK_SURFACES'--> geology + dxf
                'TOPOLOGY'--> topology
                'ANOM_FROM_BLOCK'--> just gravity and magnetics
                'ALL'--> everything

    Returns
    -------
        err code (str): error code
        files depending on the option:
            *.g00 general information
            *.g01 density
            *.g02 magnetic susceptability
            *.g12 geology rock number
            *.grv simulated gravity
            *.g20  a header file listing the a summary of the history file
             (# events, dimensions, origin and scale of block, abreviated list
              of events, complete stratigraphy )

            *.g21 block model of voxel level topology, each voxel represented
             by a string of integers (1 for each event in youngest to oldest
             order, coded as:
                BASE STRAT 0
                UNC STRAT  3
                IGNEOUS    5
                FAULT      2, 7 or 8
             so a code of 0300 shows a voxel created at time step 3 (reading
             from right, the base strat is time step 1) by an igenous event and
             of 2300 shows a voxel created at time step 3 by an igenous event
             and then to one side of a fault in time step 4 (0300 would
             therefre have been the other side of the fault)

             *.g22 The number of lithological units defined in the history

    """
    folder = os.getcwd()

    if(Windows):
        noddyEXE = folder+'/noddy.exe'
        errcode = subprocess.call(
                           [noddyEXE, historyfile, output_name, outputoption],
                           shell=False, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           creationflags=CREATE_NO_WINDOW)
    else:
        noddyEXE = folder+'/noddy_linux.exe'
        errcode = subprocess.call(
                           [noddyEXE, historyfile, output_name, outputoption],
                           shell=False, stderr=subprocess.PIPE,
                           stdout=subprocess.PIPE)

    return errcode

def simulate_calc_mismatch(P):
    """
    Simulate observations and calculate mismatch between observed and sim data.

    Parameters
    ----------
    P : dict
        A container of all the important parameters.

    Returns
    -------
    None.

    """

    # First calculate the gology and topology fields

    # There is an unresolved issue that the Noddy executable sometimes throws
    # an error, and therefore this command to run a subprocess needs to be
    # run several times until it succeeds (gets an err code of 1).
    num_tries = 0
    errcode = -1
    while((errcode != 1) & (num_tries < 100)):

        errcode = calculate_model(P['SampledInputFileName'], P['output_name'],
                                  outputoption='TOPOLOGY', Windows=P['HypP']['Windows'])
        if(P['verbose'] & (num_tries > 0)):
            print('Topology & geology calc is having issues: '
                  + str(num_tries) + ' times')
        num_tries = num_tries + 1

    if(P['verbose']):
        print('Finished calculating geology and topology')

    # If the gravity/magnetic field needs to be calculated, then need to run
    # the "ANOM_FROM_BLOCK" option of the Noddy exectuable.
    # Again, this calculate_model command sometimes needs to be run several
    # times until it actually works

    num_tries = 0
    errcode = -1
    if(('Grav' in P['DataTypes']) | ('Mag' in P['DataTypes'])):

        while((errcode != 1) & (num_tries < 100)):
            errcode = calculate_model(P['SampledInputFileName'],
                                      P['output_name'],
                                      outputoption='ANOM_FROM_BLOCK',
                                      Windows=P['HypP']['Windows'])
            if(P['verbose'] & (num_tries > 0)):
                print('Gravity & mag calc is having issues: ' + str(num_tries)
                      + ' times')
            num_tries = num_tries + 1

        if(P['verbose']):
            print('Finished calculating gravity and magnetics')

    # Calculate the gravity mismatch
    if('Grav' in P['DataTypes']):
        calc_gravity(P)

    # Calculate the magnetics mismatch
    if('Mag' in P['DataTypes']):
        calc_magnetics(P)

    # Calculate the granite top mismatch
    if('GT' in P['DataTypes']):
        calc_granitetop(P)

    # Calculate the tracer mismatch
    if('Tracer' in P['DataTypes']):
        calc_tracer(P)

    # Calculate the fault intersections
    if('FaultMarkers' in P['DataTypes']):
        calc_fault_markers(P)


def calc_gravity(P):
    """Calculate the mismatch between observed and simulated gravity data."""

    # Read in the gravity file
    output_name = P['output_name']
    filename = output_name+'.grv'
    GravityData = np.genfromtxt(filename, delimiter='\t', skip_header=8)
    GravityData = GravityData[:, 0:-1]

    simGravity = GravityData[::-1, :]
    P['Grav']['sim'] = simGravity
    
    # interpolate the gravity value at the observation points
    x = np.linspace(P['xmin'], P['xmax'], np.shape(simGravity)[1], 
                    dtype=np.float32)
    y = np.linspace(P['ymin'], P['ymax'], np.shape(simGravity)[0], 
                    dtype=np.float32)
    gSim = interpn((y,x), simGravity, 
                   np.array([P['Grav']['yObs'],P['Grav']['xObs']]).T,
                   method='linear')

    # shift the data
    gSim = P['toolbox'].shift_geophys(P, gSim, 'Grav')      
    
    # calculate error
    pt_error_array_L1 = L1_err(gSim, P['Grav']['Obs'])

    # book keeping
    book_keep(P, pt_error_array_L1, 'Grav')
    
    # remove the file to ensure that it is not accidnetally used in the
    # next iteration
    os.remove(P['output_name']+'.grv')

def median_shift(P, sim, datatype='Grav'):
    '''Center the simulated data around the observed median data'''
    med_sim = np.median(sim)
    sim = (sim-med_sim) + P[datatype]['medianObs']
    P[datatype]['simViz']=P[datatype]['sim'] - med_sim + P[datatype]['medianObs']
    
    return sim

def median_const_shift(P, sim, datatype='Grav'):
    '''Center the simulated data around the observed median data'''
    if(P['iterationNum']<50):
        med_sim = np.median(sim)
        if(P['iterationNum']==0):
            P[datatype]['med_sim'] = [med_sim]
        else:
            P[datatype]['med_sim'].append(med_sim)
    else:
        med_sim = np.median(P[datatype]['med_sim'])

    sim = (sim-med_sim) + P[datatype]['medianObs']
    P[datatype]['simViz']=P[datatype]['sim'] - med_sim + P[datatype]['medianObs']
    
    return sim

def const_shift(P, sim, datatype='Grav'):
    '''Shift the simulated data by a pre-determined amount'''

    sim = sim + P['constShift']
    P[datatype]['simViz'] = P[datatype]['sim'] - P[datatype+'ConstShift']

    return sim

def no_shift(P, gSim, datatype='Grav'):
    '''no shifts'''

    P[datatype]['simViz']=P[datatype]['sim']

    return gSim

def L1_err(sim, obs):

    mismatch = np.abs(obs-sim) 

    return mismatch


def L2_err(sim, obs):

    mismatch = (obs-sim)**2 

    return mismatch
    
def Lhalf_err(sim, obs):

    mismatch = (np.abs(obs-sim))**0.5

    return mismatch
 
def book_keep(P, pt_error_array_L1, datatype='Grav'):
    '''Save parameters to the data structures in a conssistent formatting'''

    mean_L1_err = np.mean(pt_error_array_L1)
   
    #always keep track of the L1 Norm mismatch for evalation purposes
    if(P['iterationNum']==0):
        P[datatype]['L1MismatchList'] = [mean_L1_err]
        P[datatype]['L1MismatchMatrix'] = pt_error_array_L1.reshape((-1, 1))
    else:
        P[datatype]['L1MismatchList'].append(mean_L1_err)
        P[datatype]['L1MismatchMatrix'] = np.hstack(
                                             (P[datatype]['L1MismatchMatrix'],
                                              pt_error_array_L1.reshape((-1, 1)))
                                             )

def calc_magnetics(P):
    """Calculate the mismatch between observed and simulated tracer data."""

    output_name = P['output_name']
    filename = output_name+'.mag'
    simMag = np.genfromtxt(filename, delimiter='\t', skip_header=8)
    simMag = simMag[:, 0:-1]
    simMag = simMag[::-1, :]
    P['Mag']['sim'] = simMag

    #interpolate the magnetic values at the observation points     
    x = np.linspace(P['xmin'], P['xmax'], np.shape(simMag)[1], 
                    dtype=float)
    y = np.linspace(P['ymin'], P['ymax'], np.shape(simMag)[0], 
                    dtype=np.float32)
    simMag = interpn((y,x), simMag, 
                      np.array([P['Mag']['yObs'], P['Mag']['xObs']]).T, 
                      method='linear')

    # shift the data
    simMag = P['toolbox'].shift_geophys(P, simMag, 'Mag')      
    P['gSimMag'] = simMag
    
    # calculate error
    pt_error_array_L1 = L1_err(simMag, P['Mag']['Obs'])

    # book keeping
    book_keep(P, pt_error_array_L1, 'Mag')
    
    # remove the file to ensure that it is not accidnetally used in the
    # next iteration
    os.remove(P['output_name']+'.mag')
    
def calc_granitetop(P):
    """Calculate the mismatch between observed and simulated granite top data."""

    # First determine the model dimensions from the .g00 file
    if(P['iterationNum']==0):
        get_model_dimensions(P)

    # Load the lithoology data
    output_name = P['output_name']
    #load and reshape files
    filename = output_name+'.g12'
    LithologyCodes = np.genfromtxt(filename, delimiter='\t', dtype=int)
    LithologyCodes = LithologyCodes[:, 0:-1]

    lithology = np.zeros(P['shapeL'])
    for i in range(P['shapeL'][2]):
        startIdx = P['shapeL'][1]*i 
        stopIdx = P['shapeL'][1]*(i+1)
        lithology[:,:,i] = LithologyCodes[startIdx:stopIdx,:].T
    lithology = lithology[::-1,:,:]

    # Find the first indices of the top of granite (in the z direction)
    topgraniteIdx = np.argmax(lithology==P['HypP']['graniteIdx'], axis=2) 
    topgranite = P['zmax']-topgraniteIdx*float(P['cubesize'])

    # This whole section is added because of the case of the intrusions
    # When there are instrusions, then the area around them can have 
    # unreasonable lithologies. Also the intrusions themsevelves can mean there
    # is no index for the top of granite, since the intrusions blocks the top 
    # of granite.
    # The following checks for each cell whether its neighbor is an intrusions,
    # and then if the neighbor is an intrusion, and the granite depth
    # is greater than a certain expected amount, the value of the granite top
    # for that depth is removed.
    # If there are no intrusions, this whole section can be removed.
    topgranite[(topgraniteIdx==0)|((topgraniteIdx>19))]=np.NaN
    isnanright = np.pad(topgranite,((0,0),(1,0)), mode='constant')[:, :-1]
    isnanleft = np.pad(topgranite,((0,0),(0,1)), mode='constant')[:, 1:]
    isnantop = np.pad(topgranite,((1,0),(0,0)), mode='constant')[:-1, :]
    isnanbottom = np.pad(topgranite,((0,1),(0,0)), mode='constant')[1:, :]
    isnantopright = np.pad(topgranite,((1,0),(1,0)), mode='constant')[:-1, :-1]
    isnantopleft = np.pad(topgranite,((1,0),(0,1)), mode='constant')[:-1, 1:]
    isnanbottomleft = np.pad(topgranite,((0,1),(0,1)), mode='constant')[1:, 1:]
    isnanbottomright = np.pad(topgranite,((0,1),(1,0)), mode='constant')[1:, :-1]
    isnanTotal =  (isnanright+isnanleft+isnantop+isnanbottom+isnantopright+isnantopleft+
                   isnanbottomleft+isnanbottomright)

    isnanright = np.pad(topgranite,((0,0),(2,0)), mode='constant')[:, :-2]
    isnanleft = np.pad(topgranite,((0,0),(0,2)), mode='constant')[:, 2:]
    isnantop = np.pad(topgranite,((2,0),(0,0)), mode='constant')[:-2, :]
    isnanbottom = np.pad(topgranite,((0,2),(0,0)), mode='constant')[2:, :]
    isnantopright = np.pad(topgranite,((2,0),(2,0)), mode='constant')[:-2, :-2]
    isnantopleft = np.pad(topgranite,((2,0),(0,2)), mode='constant')[:-2, 2:]
    isnanbottomleft = np.pad(topgranite,((0,2),(0,2)), mode='constant')[2:, 2:]
    isnanbottomright = np.pad(topgranite,((0,2),(2,0)), mode='constant')[2:, :-2]
    isnanTotal =  (isnanTotal+isnanright+isnanleft+isnantop+isnanbottom+isnantopright+isnantopleft+
                   isnanbottomleft+isnanbottomright)

    
    with np.errstate(invalid='ignore'):
        topgranite[((np.isnan(isnanTotal))&(topgranite<=-1200))] = np.NaN
   
    if(np.sum(topgranite<-1500)):
        print('There might be an issue')
        
    P['GT']['simViz']=topgranite
    P['xLith'] = np.linspace(P['xminL'], P['xmaxL'], P['nxL'], dtype=np.float32)+P['xmin']
    P['yLith'] = np.linspace(P['yminL'], P['ymaxL'], P['nyL'], dtype=np.float32)+P['ymin']
    P['yyLith'], P['xxLith'] = np.meshgrid(P['yLith'], P['xLith'], indexing='ij')
    #get only the valid values
    filteroutNan = ~np.isnan(topgranite) 
    x1 = P['xxLith'][filteroutNan]
    y1 = P['yyLith'][filteroutNan]
    newtopgranite = topgranite[filteroutNan]
   
    GT_Sim = interpolate.griddata((x1, y1), newtopgranite.ravel(),
                              (P['GT']['xObs'], P['GT']['yObs']), method='linear')

    # calculate error
    pt_error_array_L1 = L1_err(GT_Sim, P['GT']['Obs'])

    # book keeping
    book_keep(P, pt_error_array_L1, 'GT')


def get_model_dimensions(P):
    """Load information about model discretisation from .g00 file"""

    output_name = P['output_name']
    filename = output_name+'.g00'

    filelines = open(filename).readlines() 
    for line in filelines:
        if 'NUMBER OF LAYERS' in line:
            P['nzL'] = int(line.split("=")[1])
        elif 'LAYER 1 DIMENSIONS' in line:
            (P['nxL'], P['nyL']) = [int(l) for l in line.split("=")[1].split(" ")[1:]]
        elif 'UPPER SW CORNER' in line:
            l = [float(l) for l in line.split("=")[1].split(" ")[1:]]
            (P['xminL'], P['yminL'], P['zmaxL']) = l
        elif 'LOWER NE CORNER' in line:
            l = [float(l) for l in line.split("=")[1].split(" ")[1:]]
            (P['xmaxL'], P['ymaxL'], P['zminL']) = l
        elif 'NUM ROCK' in line:
            n_rocktypes = int(line.split('=')[1])

    P['shapeL'] = (P['nyL'], P['nxL'], P['nzL'])
    
def calc_tracer(P):
    """Calculate the mismatch between observed and simulated tracer data."""

    # The first time the simulated tracer is calculated, need to find the 
    # indices of the blocks with the well data
    if(P['iterationNum']==0):
        P['wellBlockIndices'] = getWellPathIndices(P)
    
    # Find the blocks that have faults in them
    get_fault_blocks(P)

    # The observed tracer tests that connected
    tracerconnections = P['Tracer']['Connections']

    Injectors = np.unique(tracerconnections['Injector'])    
    Producers = np.unique(tracerconnections['Producer'])    
    Wells = np.unique(np.concatenate((Injectors,Producers)))

    # Find the faults intersected by each wellbore
    connected2Faults = {}
    for i in range(len(Wells)):       
        # find the fault codes along the well paths of the injectors and 
        # producers connected in the tracer test
        # xEdge ==> fault exists between cubes along x edge at certain point
        intCodes = np.unique(np.concatenate((
                             P['xEdge'][P['wellBlockIndices'][Wells[i]]].reshape(-1,1), 
                             P['yEdge'][P['wellBlockIndices'][Wells[i]]].reshape(-1,1),
                             P['zEdge'][P['wellBlockIndices'][Wells[i]]].reshape(-1,1))))

        listC = [] 
        for j in range(len(intCodes)):
            listC = listC+P['zEdgeDict'][intCodes[j]]
        connectedFaultsi = np.unique(listC)
        connectedFaultsi = connectedFaultsi[connectedFaultsi>0]
        connected2Faults[Wells[i]] = connectedFaultsi
    
    # Initialize with zero the simulated connected tracer tests
    simTracerConnections = np.zeros((len(tracerconnections),1))
    
    # adjMatrix = fault connectivity matrix 
    adjMatrix = P['FaultMarkerMatrix'] 
    # If two faults are not connected, need to assign the distance between them
    # to infinity
    adjMatrix[adjMatrix==0] = np.Inf
    FaultConnectionMatrix = scipy.sparse.csgraph.dijkstra(adjMatrix)
    
    # Check whether the observed connections between wells is simulatedin the
    # realization ofthe gologic model
    for i in range(len(tracerconnections)):
        # check if injection well is connected to any fault
        faultConnectionsInj = connected2Faults[tracerconnections['Injector'][i]]
        if(len(faultConnectionsInj)==0):
            simTracerConnections[i] = 0
            continue

        # check if producer well is connected to any fault
        faultConnectionsPro = connected2Faults[tracerconnections['Producer'][i]]
        if(len(faultConnectionsPro)==0):
            simTracerConnections[i] = 0
            continue

        # if both are connected to a fault, then need to check if they are 
        # connected to each other
        simTracerConnections[i] = CheckFaultConnection(faultConnectionsInj, faultConnectionsPro, FaultConnectionMatrix)
            
    # The tracer observation is that all the specified wells are connected (=1)
    P['TracersConnected'] = simTracerConnections
    TracersObs = np.ones((len(P['TracersConnected']),1))

    # calculate error
    pt_error_array_L1 = L1_err(simTracerConnections, TracersObs)

    # book keeping
    book_keep(P, pt_error_array_L1, 'Tracer')
    
def getWellPathIndices(P):
    '''Get the indices of the model cubes along the well paths''' 

    uniqetracerWells = np.unique(P['WellPaths']['WellName'])
    wellBlockIndices = {}
    for i in range(len(uniqetracerWells)):
        # Get the x,y,z for each well
        welli = uniqetracerWells[i]
        wellfilter = (P['WellPaths']['WellName']==welli)
        x = P['WellPaths'].loc[wellfilter, 'Xm'].values
        y = P['WellPaths'].loc[wellfilter, 'Ym'].values
        z = P['WellPaths'].loc[wellfilter, 'Zm'].values

        # Get the distance between the xyz coordinates of the well and the 
        # model boundaries, and then calculate the block indices for each 
        # well. The lithology bounaries are often larger than the original
        # input lithology boundaries.
        xLithOrigin = P['xmin']+P['xminL']
        yLithOrigin = P['ymin']+P['yminL']
        zLithOrigin = P['zmin']+P['zminL']
        xIdx = np.floor((x-xLithOrigin)/P['cubesize']).astype(int)
        yIdx = np.floor((y-yLithOrigin)/P['cubesize']).astype(int)
        zIdx = np.floor((zLithOrigin-z)/P['cubesize']).astype(int)
        Idx = (xIdx, yIdx, zIdx)
        wellBlockIndices[welli] = Idx
    return wellBlockIndices


def get_fault_blocks(P):

    # 1. Load the topology file
    #``````````````````````````
    # This is a file where each block has a string code indicating all the 
    # geological events of different cubes
    output_name = P['output_name']
    filename = output_name+'.g21'
    EventCodesTxt = np.genfromtxt(filename, delimiter='\t', dtype=str)
    #There is some issue with the newline charchter --> loads it as well
    EventCodesTxt = EventCodesTxt[:, 0:-1]
    EventCodesTxt = np.reshape(EventCodesTxt, (P['nzL'],P['nyL'],P['nxL']))
    EventCodesTxt = np.swapaxes(EventCodesTxt,0,2)  
    EventCodesTxt=EventCodesTxt[:, :, ::-1]
    EventCodesTxt=EventCodesTxt[:, ::-1, :]

     
    # 2. Detect faults by comparing topology event codes of neighboring blocks
    #`````````````````````````````````````````````````````````````````````````
    # This is done be concatenating the topology string codes of blocks
    # with those of their neighbor blocks in the x, y, and z directions

    combine_neighbor_z = np.core.defchararray.add(EventCodesTxt[:, :, 0:-1], 
                                                  EventCodesTxt[:, :, 1:])
    zEdge = np.empty(np.shape(EventCodesTxt), dtype=combine_neighbor_z.dtype)
    zEdge[:, :, 1:]= combine_neighbor_z
    extraLayer = np.chararray((np.shape(zEdge)[0],np.shape(zEdge)[1],1))
    extraLayer = '0'*len(combine_neighbor_z[0,0,0])
    zEdge[:, :, 0]= extraLayer   
    unique_neighbor_z = np.unique(combine_neighbor_z)

    combine_neighbor_y = np.core.defchararray.add(EventCodesTxt[:, 0:-1, :],
                                                  EventCodesTxt[:, 1:, :])
    yEdge = np.empty(np.shape(EventCodesTxt), dtype=combine_neighbor_y.dtype)
    yEdge[:, :-1, :]= combine_neighbor_y
    extraLayer = np.chararray((np.shape(yEdge)[0],np.shape(yEdge)[2],1))
    extraLayer = '0'*len(combine_neighbor_y[0,0,0])
    yEdge[:, -1, :]= extraLayer
    unique_neighbor_y = np.unique(combine_neighbor_y)

    combine_neighbor_x = np.core.defchararray.add(EventCodesTxt[0:-1, :, :], 
                                                  EventCodesTxt[1:, :, :])
    xEdge = np.empty(np.shape(EventCodesTxt), dtype=combine_neighbor_x.dtype)
    xEdge[:-1, :, :]= combine_neighbor_x
    extraLayer = np.chararray((np.shape(xEdge)[1],np.shape(xEdge)[2],1))
    extraLayer = '0'*len(combine_neighbor_x[0,0,0])
    xEdge[-1, :, :]= extraLayer
    unique_neighbor_x = np.unique(combine_neighbor_x)
    
    # find all unique combinations of event codes that are adjacent to each 
    # other
    uniqueTopoNeighborCodes = np.unique(np.concatenate((unique_neighbor_x, 
                                           unique_neighbor_y, 
                                           unique_neighbor_z)))
    nEvents = int(len(combine_neighbor_z[0,0,0])/2)

    # 3. Review code list to find codes with faults and fault intersections
    #``````````````````````````````````````````````````````````````````````
    # review the unique combined codes of each cube with its neighbor, and find
    # topology codes that indicate that there is a fault between a block and 
    # its neighbor
    
    # Create an empty fault connectivity matrix (diagonal intialized with 1
    # since every fault is connected to itself)
    nFaults = getNumberFaults(output_name+'.g20')
    FaultMarkerMatrix = np.eye(nFaults)
   
    # run through the codes and select all those that have a fault in them
    # a fault is defined by two neighboring blocks with either a 7-8 contrast 
    # or a 0-2 contrast

    # faultNubers translates between the the number of the event (its location
    # in the topology string) and the number of the fault. It assumes
    # all of the faults are the final events in the history. It can also 
    # be converted to the equation: faultNum=EventNum-numbernonfaultevents
    faultNumbers = (nEvents-nFaults)*[0]+ list(np.arange(nFaults))

    listFaultMarkers=[]
    listEventsFaultMarkers=[]
    intCodeList=[]
    intCode = 0
    for i in range(len(uniqueTopoNeighborCodes)):
        codei = uniqueTopoNeighborCodes[i]

        listFaultMarkers_i=[]
        listEventsFaultMarkers_i=[]

        for k in range(nEvents):
            char1 = codei[k]
            char2 = codei[k+nEvents]
            if(char1==char2):
                continue
            else:
                if(((char1=='7')&(char2=='8'))|((char1=='8')&(char2=='7'))):
                    listFaultMarkers_i.append(codei)
                    listEventsFaultMarkers_i.append(faultNumbers[k])
                elif(((char1=='0')&(char2=='2'))|((char1=='2')&(char2=='0'))):
                    listFaultMarkers_i.append(codei)
                    listEventsFaultMarkers_i.append(faultNumbers[k])
        
        # If a certain code contains more than one instance of faulting, 
        # then we know that there is a block in the model where two faults
        # intersect, and we therefore update the fault intersection matrix
        if(len(listEventsFaultMarkers_i)>0):
            intersections = list(itertools.combinations(listEventsFaultMarkers_i, 2))
            for j in range(len(intersections)):
                fault1 = intersections[j][0]
                fault2 = intersections[j][1]
                
                FaultMarkerMatrix[fault1, fault2] = 1
                FaultMarkerMatrix[fault2, fault1] = 1
        
        # collect codes where there is a fault indicated in the code
        if(len(listEventsFaultMarkers_i)>0):               
            listFaultMarkers.append(codei)
            intCodeList.append(intCode)
            listEventsFaultMarkers.append(listEventsFaultMarkers_i)
            intCode = intCode+1
        else:
            listFaultMarkers.append(codei)
            listEventsFaultMarkers.append([-1])
            intCodeList.append(-1)
        
    # 4. determine which blocks have those fault codes in them
    #`````````````````````````````````````````````````````````
    
    # Connect between codes and fault numbers   
    dictV = dict(zip(listFaultMarkers, intCodeList)) 
    
    # Replace the codes of the blocks in the matrix with the shortened intcode
    IntCodesX = VecDict(xEdge, dictV)
    IntCodesY = VecDict(yEdge, dictV)
    IntCodesZ = VecDict(zEdge, dictV)

    # Combine all of the codes, with precedent given to codes from the x 
    # neighbor direction
    IntCodesAll = IntCodesX.copy()
    IntCodesAll[IntCodesAll<0]=IntCodesY[IntCodesAll<0]
    IntCodesAll[IntCodesAll<0]=IntCodesZ[IntCodesAll<0]
    
    # get the xyz of fault locations based on the indices of faults
    Fault3DPoints = np.nonzero(IntCodesAll)
    x = Fault3DPoints[0]*P['cubesize']+P['xmin']+P['xminL']+P['cubesize']
    y = Fault3DPoints[1]*P['cubesize']+P['ymin']+P['yminL']+P['cubesize']
    z = P['zmin']+ Fault3DPoints[2]*P['cubesize']
    P['faultNewTraces'] = np.concatenate((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)), axis=1)
    
    # 5. Some book keeping
    #``````````````
   
    P['xEdge'] = IntCodesX
    P['yEdge'] = IntCodesY
    P['zEdge'] = IntCodesZ
    P['zEdgeDict'] = dict(zip(intCodeList, listEventsFaultMarkers)) 
    P['CombinedFaultMatrix'] = IntCodesAll
    P['FaultMarkerMatrix'] = FaultMarkerMatrix

def VecDict(A, d):
    '''Substitute the values in matrix A according to the dictionary d'''
    
    return np.vectorize(d.__getitem__)(A)
  
def getNumberFaults(file):
    '''Get the number of faults based on the .g00 file'''
    
    file1 = open(file, 'r') 
    Lines = file1.readlines() 
      
    # Strips the newline character 
    lineNum = 0
    EventTypes = []
    nFaults = 0
    for line in Lines: 
        if(lineNum==0):
            val = line[10:]
            numEvents = int(val)
    
        if((lineNum>2)&(lineNum<3+numEvents)):
            val = int(line.split()[1])   
            EventTypes.append(val)
            if(val==2):
                nFaults=nFaults+1
        lineNum = lineNum+1

    return nFaults


def CheckFaultConnection(faultConnectionsInj, faultConnectionsPro, FaultConnectionMatrix):
    '''Check if an injector is connected to a producer by checking whether any
    of the faults to which the injector (faultConnectionsInj) and producer
    (faultConnectionsPro) are connected, are connected to each other'''
    
    connected=0
    for i in range(len(faultConnectionsInj)):
        f1=  faultConnectionsInj[i]
        for j in range(len(faultConnectionsPro)):
            f2 = faultConnectionsPro[j]
            c = FaultConnectionMatrix[f1, f2]
            if(~np.isinf(c)):
                connected = 1
                return connected

    return connected

def calc_fault_markers(P):
    """Calculate mismatch between observed and simulated fault marker data."""

    obsFaultMarkers = P['FaultMarkers']['Obs']
    wellpaths = P['FaultMarkers']['WellData']

    # Initialize simulated fault markers array
    simDistance2Markers = np.zeros((len(obsFaultMarkers),1))

    wells = np.unique(obsFaultMarkers['WellName'])

    # Create an interpolation object connecting xyz coordinates with the fualt
    # code
    z = (np.linspace(P['zminL'], P['zmaxL'], P['nzL'], dtype=np.float32)
        +P['zmin'])
    faultinterpolator = RegularGridInterpolator((P['xLith'], P['yLith'], z), 
                                            P['xEdge']+P['yEdge']+P['zEdge'], 
                                            method='nearest')
    
   
    intersectionsListx = []
    intersectionsListy = []
    intersectionsListz = []
    idList = []
    for i in range(len(wells)):
        filterWell = wellpaths['WellName']==wells[i]

        #get fault data along wellbore        
        wellpath = wellpaths.loc[filterWell, ['Xm','Ym','Zm']].values                                
        faultdataalongwell = faultinterpolator(wellpath)
        intersectionpoints = wellpath[faultdataalongwell>0,:]
        intersectionsListx = intersectionsListx + list(intersectionpoints[:,0])
        intersectionsListy = intersectionsListy + list(intersectionpoints[:,1])
        intersectionsListz = intersectionsListz + list(intersectionpoints[:,2])
        idList = idList + [i]*len(intersectionpoints)
              
        #get fault data at marker location
        filtermarkers = obsFaultMarkers['WellName']==wells[i]
        markerpositions = obsFaultMarkers.loc[filtermarkers, ['X', 'Y', 'Z']].values                 

        # The distance between the observed and simulated markers, is the 
        # distance between the observed marker and the nearest simulated marker
        # If the distance is greater than a P['MaxFaultMarkerError'], or
        # if there is not fault intersecting the wellbore, then the distance is
        # equal to P['MaxFaultMarkerError']
        markers = markerpositions[:, 2]
        distances = np.zeros((len(markers),1))
        for m in range(len(markers)):
            if(len(intersectionpoints)==0):
                distances[m]=P['HypP']['MaxFaultMarkerError']
            else:
                v=np.min(np.abs(markers[m]-intersectionpoints[:, 2]))
                distances[m] = np.min([v, P['HypP']['MaxFaultMarkerError']])
        simDistance2Markers[filtermarkers]=distances

    # The marker observation is that all the markers have a zero distance to 
    # themselves
    MarkersObs = np.zeros((len(simDistance2Markers),1))

    # calculate error and normalize by the confidence of each pick/marker
    confidenceM = P['FaultMarkers']['Obs']['Confidence'].values.reshape((-1,1))
    pt_error_array_L1 = np.abs(simDistance2Markers - MarkersObs)
    pt_error_array_L1 = (pt_error_array_L1*confidenceM)/np.mean(confidenceM)

    # book keeping
    book_keep(P, pt_error_array_L1, 'FaultMarkers')
    P['simDistance2Markers'] =simDistance2Markers
    P['FaultMarkers']['simX'] = intersectionsListx
    P['FaultMarkers']['simY'] = intersectionsListy
    P['FaultMarkers']['simZ'] = intersectionsListz
    P['FaultMarkers']['simID'] = idList      