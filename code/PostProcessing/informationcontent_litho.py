# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:32:25 2020

@author: ahinoamp

This script lookps through pickle files of the lithology blocks and calculates
the lithology block prabilities and the information entropy

"""

import pandas as pd
import numpy as np
from glob import glob
import pickle
import math

folder = 'Z:/TransferThesisDraft/HistoryFileTransfer/Lithology/'

Norm = pd.read_pickle('NormalizingFactorSummary.pkl')

picklefiles = glob(folder+'*.pickle')
nFiles = len(picklefiles)

started=0
numPrior = nFiles
ErrList=[]
n_models_pri = 0  # number of models loaded
n_models_post = 0  # number of models loaded
litholist=[1, 2, 3, 4, 5, 6, 7]
#for i in range(500):  
nx = 74
ny = 76
nz = 26
p_block_pri = {}
p_block_post = {}

for i in litholist:
    p_block_pri[i] = np.ndarray((nx, ny, nz))
    p_block_post[i] = np.ndarray((nx, ny, nz))


for i in range(nFiles):    
#for i in range(50):    

    print('reading: '+picklefiles[i])

    with open(picklefiles[i], 'rb') as handle:
        dictFaultI = pickle.load(handle)
        
    FaultPriorMatrix = dictFaultI['Lithology']
    Err = dictFaultI['LithologyErr']
    NormErr = (Err[0]/Norm['Grav'] + 
               Err[1]/Norm['Mag'] +
               Err[2]/Norm['Tracer'] +
               Err[3]/Norm['GT'] +
               Err[4]/Norm['FaultIntersection'])/5.0
    
    ErrList.append(NormErr)    

    #prior stuff


    # loop through voxels and tally frequencies
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # get litho
                litho = int(FaultPriorMatrix[x][y][z]) 
                # update litho frequency
                p_block_pri[litho][x][y][z] += 1

    # keep track of the number of models we've loaded
    n_models_pri += 1

    #posterior stuff

    if(NormErr<0.65):                              
        # loop through voxels and tally frequencies
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    # get litho
                    litho = int(FaultPriorMatrix[x][y][z]) 
                    # update litho frequency
                    p_block_post[litho][x][y][z] += 1
    
        # keep track of the number of models we've loaded
        n_models_post += 1

    
 
    #prior stuff

    # convert frequency fields to probabilities & calculate information entropy
e_block_pri = np.ndarray((nx, ny, nz))
for x in range(nx):
    for y in range(ny):
        for z in range(nz):
            entropy = 0.0
            for litho in litholist:
                # convert frequency to probability
                p_block_pri[litho][x][y][z] = p_block_pri[litho][x][y][z] / float(n_models_pri)

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


#posterior stuff
e_block_post = np.ndarray((nx, ny, nz))
# convert frequency fields to probabilities & calculate information entropy
for x in range(nx):
    for y in range(ny):
        for z in range(nz):
            entropy = 0.0
            for litho in litholist:
                # convert frequency to probability
                p_block_post[litho][x][y][z] = p_block_post[litho][x][y][z] / float(n_models_post)

                # fix domain to 0 < p < 1
                if(p_block_post[litho][x][y][z] == 0):
                    p_block_post[litho][x][y][z] = 0.0000000000000001
                if(p_block_post[litho][x][y][z] >= 0.9999999999999999):
                    p_block_post[litho][x][y][z] = 0.9999999999999999

                # calculate
                p = p_block_post[litho][x][y][z]  # shorthand
                entropy += p * math.log(p, 2) + (1 - p) * (math.log(1 - p, 2))

            # entropy = entropy * -1 / float(self.n_rocktypes) #divide by n
            e_block_post[x][y][z] = entropy

np.savetxt(folder+'e_block_pri_litho.csv', -1*e_block_pri.ravel(), delimiter=',', fmt='%.4f') 
np.savetxt(folder+'e_block_post_litho.csv', -1*e_block_post.ravel(), delimiter=',', fmt='%.4f') 
print(np.min(e_block_pri))
print(np.min(e_block_post))
