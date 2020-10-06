# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:13:56 2020

@author: ahinoamp

Convert a lithology block to csv

"""
import pynoddy
import vtkplotter as vtkP
import time
import numpy as np
import meshio
import pandas as pd
import matplotlib.pylab as pl
from scipy.interpolate import griddata
import GravityInversionUtilities as GI
from scipy.interpolate import RegularGridInterpolator


xy_origin=[317883,4379246, 1200-4000]
xy_extent = [9000,9400,4000]
xy_origin_o = np.asarray([316383, 4377746,   -4300])-100
xy_extent_o = np.asarray([12000, 12400,  5500])
xy_extremity_o = np.asarray([328383, 4390146,    1200])+100

output_name = 'ScratchPlots/ThreeD_Plots/vtkscratch'

N1 = pynoddy.output.NoddyOutput(output_name)
Lithology = N1.block
#    Lithology=np.swapaxes(Lithology,0,2)

x = np.linspace(N1.xmin, N1.xmax, N1.nx, dtype=np.float32)+xy_origin[0]
y = np.linspace(N1.ymin, N1.ymax, N1.ny, dtype=np.float32)+xy_origin[1]
z = np.linspace(N1.zmin, N1.zmax, N1.nz, dtype=np.float32)+xy_origin[2]
x[0] = xy_origin_o[0]
y[0] = xy_origin_o[1]
z[0] = xy_origin_o[2]
x[-1] = xy_extremity_o[0]
y[-1] = xy_extremity_o[1]
z[-1] = xy_extremity_o[2]

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

xlist = xx.reshape(-1,)
ylist = yy.reshape(-1,)
zlist = zz.reshape(-1,)
llist = Lithology.reshape(-1,)

lithblock = pd.DataFrame({'x': xlist,
                          'y': ylist,
                          'z': zlist,
                          'lithrock':llist})
lithblock.to_csv('lithology.csv', index=False)

xExtrap = np.arange(xy_origin_o[0], xy_extremity_o[0], 45, dtype=np.float32)
yExtrap = np.arange(xy_origin_o[1], xy_extremity_o[1], 45, dtype=np.float32)
zExtrap = np.arange(xy_origin_o[2], xy_extremity_o[2], 45, dtype=np.float32)

xxExtrap, yyExtrap, zzExtrap = np.meshgrid(xExtrap, yExtrap, zExtrap, indexing='ij')

xlist = xxExtrap.reshape(-1,1)
ylist = yyExtrap.reshape(-1,1)
zlist = zzExtrap.reshape(-1,1)
points = np.concatenate((xlist, ylist, zlist), axis=1)

faultdatainterpolator = RegularGridInterpolator((x, y, z), Lithology, method='nearest')
Lithinterp = faultdatainterpolator(points)

lithblock = pd.DataFrame({'x': xlist.reshape(-1,),
                          'y': ylist.reshape(-1,),
                          'z': zlist.reshape(-1,),
                          'lithrock':Lithinterp.reshape(-1,)})
lithblock.to_csv('lithologyE.csv', index=False)

