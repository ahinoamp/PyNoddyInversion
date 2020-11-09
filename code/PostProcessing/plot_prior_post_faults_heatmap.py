# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:52:30 2020

@author: ahinoamp
"""

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from matplotlib.lines import Line2D
import pickle

P={}
xy_origin=[316448, 4379166, -2700]
xy_extent = [8850, 9035,3900]
P['xy_origin']=xy_origin
P['xy_extent'] = xy_extent
cube_size = 150

xsection = P['xy_extent'][0]/2.0 
ysection = P['xy_extent'][1]/2.0 
zsection = 0 
        
fz = 14


folderPost = 'Scratch2/Blocks/'
folderPri = 'PickleResults/Blocks/'


Norm = {}
Norm['Grav'] = 2.4
Norm['Tracer'] = 1.0
Norm['FaultMarkers'] = 500
Norm['GT'] = 315
Norm['Mag'] = 300

###############################
## First put in prior            
###############################
picklefiles = glob(folderPri+'*.pickle')
nFiles = len(picklefiles)

fig, ax = plt.subplots(1, 1, figsize=(8,8))
started=0
started2=0
numPosterior=0
numPrior = 0
for i in range(nFiles):    
#for i in range(200):    

    print('reading: '+picklefiles[i])

    try:
        with open(picklefiles[i], 'rb') as handle:
            dictFaultI = pickle.load(handle)
    except:
        print('There was an issue loading')
        
    FaultPriorMatrix = (dictFaultI['FaultBlock']>-1).astype(int)

    if(started2==0):
        sumV = FaultPriorMatrix
        started2=1
    else:
        sumV = sumV+ FaultPriorMatrix
    numPrior=numPrior+1

###############################
## Second posterior            
###############################
picklefiles = glob(folderPost+'*.pickle')
nFiles = len(picklefiles)

for i in range(nFiles):    
#for i in range(200):    

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
        
       
    if(NormErr<0.511):
        if(started==0):
            sumPosterior = FaultPriorMatrix
            started=1
        else:
            sumPosterior= sumPosterior +FaultPriorMatrix
        numPosterior=numPosterior+1


print('numPost:' +str(numPosterior))

shapeCube = np.shape(sumV)
sliceConstX = int((xsection +1200-cube_size)/cube_size)
sliceConstY = int((ysection +1200-cube_size)/cube_size)
sliceConstZ = int((zsection+2700-cube_size)/float(cube_size))

sumV = sumV/numPrior
sumPosterior = sumPosterior/numPosterior

plt.close('all')

xdim = shapeCube[0]
ydim = shapeCube[1]
zdim = shapeCube[2]

xmin = P['xy_origin'][0]-1200
xmax = P['xy_origin'][0]+P['xy_extent'][0]+1200

ymin = P['xy_origin'][1]-1200
ymax = P['xy_origin'][1]+P['xy_extent'][1]+1200

zmin = P['xy_origin'][2]
zmax = P['xy_origin'][2]+P['xy_extent'][2]


xminV = xmin+1200
xmaxV = xmax-1200

yminV = ymin+1200
ymaxV = ymax-1200

#Slice X
slicePrior = sumV[sliceConstX, :, :].reshape(ydim, zdim)
slicePrior = np.flipud(np.transpose(slicePrior))

slicePosterior = sumPosterior[sliceConstX, :, :].reshape(ydim, zdim)
slicePosterior = np.flipud(np.transpose(slicePosterior))

vmax = np.max([np.max(slicePrior), np.max(slicePosterior)])

fig, axs = plt.subplots(1, 2, figsize=(12,6))
ax=axs[0]
ax.imshow(slicePrior, origin = 'upper',  extent = [ymin, ymax, zmin, zmax], vmin=0, vmax=vmax)
ax.set_title('Fault prior probability cross section (x='+str(xsection)+')', fontsize=fz)
ax.set_xlabel('Y (m)', fontsize=fz)
ax.set_ylabel('Z (m)', fontsize=fz)
ax.set_xlim([yminV, ymaxV])
        
ax=axs[1]
im = ax.imshow(slicePosterior, origin = 'upper', extent = [ymin, ymax, zmin, zmax], vmin=0, vmax=vmax )
ax.set_title('Fault posterior probability (x = '+str(ysection)+') cross section', fontsize=fz)
ax.set_aspect('equal')
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
ax.set_xlabel('Y (m)', fontsize=fz)
#ax.set_ylabel('Z (m)', fontsize=fz)
ax.set_yticks([])
ax.set_xlim([yminV, ymaxV])
fig.savefig('SliceX.png', dpi = 300,bbox_inches="tight")

#Slice Y
slicePrior = sumV[:, sliceConstY, :].reshape(xdim, zdim)
slicePrior = np.flipud(np.transpose(slicePrior))

slicePosterior = sumPosterior[:, sliceConstY, :].reshape(xdim, zdim)
slicePosterior = np.flipud(np.transpose(slicePosterior))

vmax = np.max([np.max(slicePrior), np.max(slicePosterior)])

fig, axs = plt.subplots(1, 2, figsize=(12,6))
ax=axs[0]
ax.imshow(slicePrior, origin = 'upper',  extent = [xmin, xmax, zmin, zmax], vmin=0, vmax=vmax)
ax.set_title('Fault prior probability (y = '+str(ysection)+') cross section', fontsize=fz)
ax.set_xlabel('X (m)', fontsize=fz)
ax.set_ylabel('Z (m)', fontsize=fz)
ax.set_xticks([317000, 319000, 321000, 323000, 325000])
ax.set_xlim([xminV, xmaxV])

ax=axs[1]
im = ax.imshow(slicePosterior, origin = 'upper', extent = [xmin, xmax, zmin, zmax], vmin=0, vmax=vmax )
ax.set_title('Fault posterior probability (y = '+str(ysection)+') cross section', fontsize=fz)
ax.set_aspect('equal')
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
ax.set_xlabel('X (m)', fontsize=fz)
#ax.set_ylabel('Z (m)', fontsize=fz)
ax.set_yticks([])
ax.set_xticks([317000, 319000, 321000, 323000, 325000])
ax.set_xlim([xminV, xmaxV])

fig.savefig('SliceY.png', dpi = 300,bbox_inches="tight")

#Slice Z
slicePrior = sumV[:, :, sliceConstZ].reshape(xdim, ydim)
slicePrior = np.transpose(np.fliplr(slicePrior))

slicePosterior = sumPosterior[:, :, sliceConstZ].reshape(xdim, ydim)
slicePosterior = np.transpose(np.fliplr(slicePosterior))

vmax = np.max([np.max(slicePrior), np.max(slicePosterior)])

fig, axs = plt.subplots(1, 2, figsize=(12,6))
plt.subplots_adjust(wspace = 0.1)
ax=axs[0]
ax.imshow(slicePrior, origin = 'upper',  extent = [xmin, xmax, ymin, ymax], vmin=0, vmax=vmax)
ax.set_title('Fault prior probability (z = '+str(zsection)+') cross section', fontsize=fz)
ax.set_xlabel('X (m)', fontsize=fz)
ax.set_ylabel('Y (m)', fontsize=fz)
ax.set_xticks([317000, 319000, 321000, 323000, 325000])
ax.set_xlim([xminV, xmaxV])
ax.set_ylim([yminV, ymaxV])

ax=axs[1]
im = ax.imshow(slicePosterior, origin = 'upper', extent = [xmin, xmax, ymin, ymax], vmin=0, vmax=vmax )
ax.set_title('Fault posterior probability (z = '+str(zsection)+') cross section', fontsize=fz)
ax.set_aspect('equal')
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax) # Similar to fig.colorbar(im, cax = cax)
ax.set_xlabel('X (m)', fontsize=fz)
#ax.set_ylabel('Y (m)', fontsize=fz)
ax.set_xticks([317000, 319000, 321000, 323000, 325000])
ax.set_yticks([])
ax.set_xlim([xminV, xmaxV])
ax.set_ylim([yminV, ymaxV])
fig.savefig('SliceZ.png', dpi = 300,bbox_inches="tight")