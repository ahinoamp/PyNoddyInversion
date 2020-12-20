# -*- coding: utf-8 -*-
"""
Created on Oct 4, 2020

@author: ahinoamp@gmail.com

This scripts uses vedo to plot the prior and posterior blocks with different
probability thresholds

"""


"""Use a variant of the Moving Least Squares (MLS)
algorithm to project a cloud of points to become a smooth surface.
In the second window we show the error estimated for
each point in color scale (left) or in size scale (right).
"""
from vedo import *
import numpy as np
import matplotlib.pyplot as plt


folder = 'C:/Users/ahino/Documents/FinalGeothermicsToday/'

scalar_field_pri = np.loadtxt(folder+'p_block_pri.csv',delimiter=',')
scalar_field_pri = np.reshape(scalar_field_pri, (74,76,26)).T
scalar_field_pri = scalar_field_pri[:,8:-8, 8:-8]

scalar_field_post = np.loadtxt(folder+'p_block_post.csv',delimiter=',')
scalar_field_post = np.reshape(scalar_field_post, (74,76,26)).T
scalar_field_post = scalar_field_post[:,8:-8, 8:-8]


spacing=[150, 150,150]
        
origin=[316448, 4379166, -2700]
extent = [150*59, 150*60, 150*26]
xmax = origin[0]+extent[0]
zmax =origin[2]+extent[2]

minV = 0.25
maxV = np.max([np.max(scalar_field_pri.ravel()), np.max(scalar_field_post.ravel())])
minVr = np.min([np.min(scalar_field_pri.ravel()), np.min(scalar_field_post.ravel())])


volPri = Volume(scalar_field_pri,spacing = spacing, origin=origin)
legoPri = volPri.legosurface(vmin=minV, vmax=maxV, cmap="viridis")
#legoPri.addScalarBar3D(title='P(fault)',
#                       titleXOffset = 2,
#                       titleRotation = 180)

volPost = Volume(scalar_field_post, spacing = spacing, origin=origin)
legoPost = volPost.legosurface(vmin=minV, vmax=maxV, cmap="viridis")
legoPost.addScalarBar3D(title='P(fault)',
                        titleXOffset = 2,
                       titleRotation = 180).z(-9000)

vp2 = Plotter(axes=dict(xtitle='Easting (m)',
                        ytitle='Northing (m)', 
                        ztitle='Elevation (m)', yzGrid=False),
                        pos=(300, 400), N=2)

cornerptsP = [[origin[0], origin[1], origin[2]],
              [origin[0]+extent[0], origin[1], origin[2]],
              [origin[0], origin[1]+extent[1], origin[2]],
              [origin[0], origin[1], origin[2]+extent[2]]]
cornerpts = Points(cornerptsP, r=0.05).c("k")

vp2.show(legoPri, cornerpts, 'hi1', at=0, elevation=-20, azimuth=0)
vp2.show(legoPost, cornerpts, at=1, interactive=1, elevation=-20, azimuth=0)

# fig, axs = plt.subplots(1, 2, figsize=(8,8))
# axs[0].hist(scalar_field_pri.ravel())
# axs[1].hist(scalar_field_post.ravel())
# fig.savefig('hh.png', dpi = 300,bbox_inches="tight")
                        