# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:32:25 2020

@author: ahinoamp@gmail.com

This script lookps through pickle files of the fault blocks and calculates
the fault block prabilities and the information entropy

"""

import pandas as pd
import numpy as np
from glob import glob
import pickle
import math

folder = 'PickleResults/Blocks/'

Norm = {}
Norm['Grav'] = 2.4
Norm['Tracer'] = 1.0
Norm['FaultMarkers'] = 500
Norm['GT'] = 315
Norm['Mag'] = 300

picklefiles = glob(folder+'*.pickle')
nFiles = len(picklefiles)

started=0
numPrior = nFiles
ErrList=[]
n_models_pri = 0  # number of models loaded
n_models_post = 0  # number of models loaded
litholist=[0,1]
#for i in range(500):  
nx = 74
ny = 76
nz = 26
n_block_pri = {}
n_block_pri[0] = np.ndarray((nx, ny, nz))
n_block_pri[1] = np.ndarray((nx, ny, nz))
n_block_post = {}
n_block_post[0] = np.ndarray((nx, ny, nz))
n_block_post[1] = np.ndarray((nx, ny, nz))

p_block_pri = {}
p_block_pri[0] = np.ndarray((nx, ny, nz))
p_block_pri[1] = np.ndarray((nx, ny, nz))
p_block_post = {}
p_block_post[0] = np.ndarray((nx, ny, nz))
p_block_post[1] = np.ndarray((nx, ny, nz))

for i in range(nFiles):    
#for i in range(500):    

    print('reading: '+picklefiles[i])

    with open(picklefiles[i], 'rb') as handle:
        dictFaultI = pickle.load(handle)
        
    FaultPriorMatrix = (dictFaultI['FaultBlock']>-1).astype(int)
    Err = dictFaultI['FaultBlockErr']
    NormErr = (Err[0]/Norm['Grav'] + 
               Err[1]/Norm['Mag'] +
               Err[2]/Norm['Tracer'] +
               Err[3]/Norm['GT'] +
               Err[4]/Norm['FaultMarkers'])/5.0
    
    ErrList.append(NormErr)    

    #prior stuff


    # loop through voxels and tally frequencies
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # get litho
                litho = int(FaultPriorMatrix[x][y][z]) 
                # update litho frequency
                n_block_pri[litho][x][y][z] += 1

    # keep track of the number of models we've loaded
    n_models_pri += 1

    #posterior stuff
    if(NormErr<0.511):                                                           
        # loop through voxels and tally frequencies
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # get litho
                    litho = int(FaultPriorMatrix[x][y][z]) 
                    # update litho frequency
                    n_block_post[litho][x][y][z] += 1
    
        # keep track of the number of models we've loaded
        n_models_post += 1

    
 
    #prior stuff

    # convert frequency fields to probabilities & calculate information entropy
e_block_pri = np.ndarray((nx, ny, nz))
for x in range(nx):
    for y in range(ny):
        for z in range(nz):
            entropy = 0.0
            for litho in range(len(litholist)):
                # convert frequency to probability
                p_block_pri[litho][x][y][z] = n_block_pri[litho][x][y][z] / float(n_models_pri)

                # fix domain to 0 < p < 1
                if(p_block_pri[litho][x][y][z] == 0):
                    p_block_pri[litho][x][y][z] = 0.0000000000000001
                if(p_block_pri[litho][x][y][z] >= 0.9999999999999999):
                    p_block_pri[litho][x][y][z] = 0.9999999999999999

                # calculate
                p = p_block_pri[litho][x][y][z]  # shorthand
                entropy += p * math.log(p, 2) + (1 - p) * (math.log(1 - p, 2))

            # entropy = entropy * -1 / float(self.n_rocktypes) #divide by n
            e_block_pri[x][y][z] = entropy


# #posterior stuff
# e_block_post = np.ndarray((nx, ny, nz))
# # convert frequency fields to probabilities & calculate information entropy
# for x in range(nx):
#     for y in range(ny):
#         for z in range(nz):
#             entropy = 0.0
#             for litho in range(len(litholist)):
#                 # convert frequency to probability
#                 p_block_post[litho][x][y][z] = n_block_post[litho][x][y][z] / float(n_models_post)

#                 # fix domain to 0 < p < 1
#                 if(p_block_post[litho][x][y][z] == 0):
#                     p_block_post[litho][x][y][z] = 0.0000000000000001
#                 if(p_block_post[litho][x][y][z] >= 0.9999999999999999):
#                     p_block_post[litho][x][y][z] = 0.9999999999999999

#                 # calculate
#                 p = p_block_post[litho][x][y][z]  # shorthand
#                 entropy += p * math.log(p, 2) + (1 - p) * (math.log(1 - p, 2))

#             # entropy = entropy * -1 / float(self.n_rocktypes) #divide by n
#             e_block_post[x][y][z] = entropy

#np.savetxt(folder+'p_block_post.csv', p_block_post[1].ravel(), delimiter=',', fmt='%.4f') 
np.savetxt('p_block_pri.csv', p_block_pri[1].ravel(), delimiter=',', fmt='%.4f') 
np.savetxt('e_block_pri.csv', -1*e_block_pri.ravel(), delimiter=',', fmt='%.4f') 
#np.savetxt(folder+'e_block_post.csv', -1*e_block_post.ravel(), delimiter=',', fmt='%.4f') 
print(np.max(e_block_pri))
#print(np.max(e_block_post))
