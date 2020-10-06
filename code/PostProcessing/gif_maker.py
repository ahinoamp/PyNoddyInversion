# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:24:18 2020

@author: ahinoamp@gmail.com

This script takes a folder and makes a gif by sorting all of the png images 
in the folder and stitching them together.
"""

from PIL import Image
import glob
from tkinter import Tcl

# Create the frames
frames = []
folder = 'PlotFrontier/'
imgs = sorted(glob.glob(folder+"*.png"))
imgs2 = Tcl().call('lsort', '-dict', imgs)

for i in imgs2:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save(folder+'png_to_gif.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300, loop=1)