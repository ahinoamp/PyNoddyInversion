# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:52:30 2020

@author: ahinoamp

Plot prior vs. posterior fault lineaments
"""
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import pickle

def getXZ(folder):
    
    picklefiles = glob(folder+'*.pickle')
    nFiles = len(picklefiles)
    print(nFiles)
    fz = 18
    
    directions = ['X', 'Y', 'Z']
    xLargePost = {}
    xLargePri = {}
    zLargePost = {}
    zLargePri = {}
    #    for i in range(200):  
    postStarted = 0    
    priStarted=0
    for i in range(nFiles):    
        print(i)
        file=picklefiles[i]
        print(file)
        with open(file, 'rb') as handle:
            dicti = pickle.load(handle)
        
        keysdicti = list(dicti.keys())
        nFaults = dicti['nFaults']
            
        for d in range(len(directions)):
    
            direction = directions[d]
    
            for j in range(nFaults):
                if(str(j) in keysdicti):
                    smalldict = dicti[str(j)]
                    keysdictii = list(smalldict.keys())
                    if(direction in keysdictii):
                        x,z = smalldict[direction]
                        if(direction not in list(xLargePost.keys())):  
                            xLargePost[direction] = x.reshape(-1,1)
                            zLargePost[direction] = z.reshape(-1,1)
                        else:
                            xLargePost[direction] = np.concatenate((xLargePost[direction], x.reshape(-1,1)), axis=1)
                            zLargePost[direction] = np.concatenate((zLargePost[direction], z.reshape(-1,1)), axis=1)

    return xLargePost, zLargePost

def getXZPost(folder,threshold, Norm):
    
    picklefiles = glob(folder+'*.pickle')
    nFiles = len(picklefiles)
    print(nFiles)
    fz = 18
    
    directions = ['X', 'Y', 'Z']
    xLargePost = {}
    xLargePri = {}
    zLargePost = {}
    zLargePri = {}
    #    for i in range(200):  
    postStarted = 0    
    priStarted=0
    for i in range(nFiles):    
        print(i)
        file=picklefiles[i]
        print(file)
        with open(file, 'rb') as handle:
            dicti = pickle.load(handle)
        
        keysdicti = list(dicti.keys())
        nFaults = dicti['nFaults']
        Err = dicti['Err']
        Err = (Err[0]/Norm['Grav'] + 
               Err[1]/Norm['Mag'] +
               Err[2]/Norm['Tracer'] +
               Err[3]/Norm['GT'] +
               Err[4]/Norm['FaultMarkers'])/5.0 

        print(Err)
        if(Err<threshold):            
            for d in range(len(directions)):
        
                direction = directions[d]
        
                for j in range(nFaults):
                    if(str(j) in keysdicti):
                        smalldict = dicti[str(j)]
                        keysdictii = list(smalldict.keys())
                        if(direction in keysdictii):
                            x,z = smalldict[direction]
                            if(direction not in list(xLargePost.keys())):  
                                xLargePost[direction] = x.reshape(-1,1)
                                zLargePost[direction] = z.reshape(-1,1)
                            else:
                                xLargePost[direction] = np.concatenate((xLargePost[direction], x.reshape(-1,1)), axis=1)
                                zLargePost[direction] = np.concatenate((zLargePost[direction], z.reshape(-1,1)), axis=1)

    return xLargePost, zLargePost

plt.close('all')

Norm = {}
Norm['Grav'] = 2.4
Norm['Tracer'] = 1.0
Norm['FaultMarkers'] = 500
Norm['GT'] = 315
Norm['Mag'] = 300
            
P={}
xy_origin=[316448, 4379166, -2700]
xy_extent = [8850, 9035,3900]
P['xy_origin']=xy_origin
P['xy_extent'] = xy_extent

folderPri = 'PickleResults/Faults/'
folderPost = 'Scratch2/Faults/'

xLargePost, zLargePost = getXZPost(folderPost, threshold=0.51,Norm=Norm)

#xLargePri, zLargePri = getXZ(folderPri)

OneLargeDict = pd.DataFrame({'xLargePost':xLargePost, 
                'zLargePost':zLargePost})

print(len(xLargePost))

#OneLargeDict = pd.DataFrame({'xLargePost':xLargePost, 
#                'zLargePost':zLargePost,
#                'xLargePri':xLargePri,
#                'zLargePri':zLargePri})

OneLargeDict.to_pickle('PosteriorFaultLineaments.pkl')