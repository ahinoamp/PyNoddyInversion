# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:53:20 2020

@author: ahinoamp
"""
import pynoddy
import vedo as vtkP
import time
import numpy as np
import meshio
import pandas as pd
import matplotlib.pylab as pl
from scipy.interpolate import griddata
import SimulationUtilities as sim
from glob import glob

def getDXF_parsed_structure(output_name):
    filename = output_name + '.dxf'
#    doc = ezdxf.readfile(filename)
    cell_data = []
    xpoint = []
    ypoint = []
    zpoint = []
    with open(filename) as f:
        cntr=0
        faceCounter=0
        for line in f:
            if(cntr==(7+faceCounter*28)):
                cell_data.append(line)
                faceCounter=faceCounter+1
            elif(cntr==(9+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(11+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(13+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(15+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(17+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(19+(faceCounter-1)*28)):
                zpoint.append(float(line))

            elif(cntr==(21+(faceCounter-1)*28)):
                xpoint.append(float(line))
            elif(cntr==(23+(faceCounter-1)*28)):
                ypoint.append(float(line))
            elif(cntr==(25+(faceCounter-1)*28)):
                zpoint.append(float(line))

            cntr=cntr+1

    points = np.column_stack((np.asarray(xpoint, dtype=float),
                             np.asarray(ypoint, dtype=float),
                             np.asarray(zpoint, dtype=float)))
    cell_data.pop()
    cell_data = np.asarray(cell_data, dtype=object)
#    print('Finished reading model')
   
#    Data = pd.DataFrame({'x': np.asarray(xpoint, dtype=float), 
#                         'y': np.asarray(ypoint, dtype=float),
#                         'z': np.asarray(zpoint, dtype=float),
#                         'cell_data': np.repeat(np.asarray(cell_data),3)})
    
#    Data.to_csv('xyz_strat.csv')
    return points, cell_data, faceCounter

def convertSurfaces2VTK(points, cell_data, faceCounter, outputOption = 1, fileprefix='Surface',  xy_origin=[0,0,0], num=0):
    
    # Choose output option
    num3Dfaces=faceCounter
    print('The number of triangle elements (cells/faces) is: ' + str(num3Dfaces))


    #apply origin transformation
    points[:, 0] = points[:, 0]+xy_origin[0]
    points[:, 1] = points[:, 1]+xy_origin[1]
    points[:, 2] = points[:, 2]+xy_origin[2]
    
    cell_data = pd.Series(cell_data.reshape((-1, )))

    CatCodes = np.zeros((len(cell_data),))
    filterB = (cell_data.str.contains('B')) 
    filterS = (cell_data.str.contains('S')) 

    CatCodes[filterB]= cell_data.loc[filterB].str[:-20].astype('category').cat.codes
    CatCodes[filterS]= -1*(cell_data.loc[filterS].str[:-12].astype('category').cat.codes+1)

    for i in range(1, len(CatCodes)):
        if(CatCodes[i]==0):
            CatCodes[i]=CatCodes[i-1]
            if(CatCodes[i-1]==0):
                CatCodes[i]=CatCodes[np.nonzero(CatCodes)[0][0]]

    UniqueCodes = np.unique(CatCodes)
    nSurfaces = len(UniqueCodes)

    Data = pd.DataFrame({'x': np.asarray(points[:, 0], dtype=float), 
                         'y': np.asarray(points[:, 1], dtype=float),
                         'z': np.asarray(points[:, 2], dtype=float),
                         'cell_data': np.repeat(np.asarray(CatCodes),3)})

    Data.to_csv('Example_Resultant_Models/150mResDataPoints/xyz_'+str(num)+'.csv')
    ## if you would like a single vtk file
    if (outputOption==2): 
        cells = np.zeros((num3Dfaces, 3),dtype ='int')
        i=0
        for f in range(num3Dfaces):
            cells[f,:]= [i, i+1, i+2]
            i=i+3
        meshio.write_points_cells(
            "Model.vtk",
            points,
            cells={'triangle':cells},
            cell_data= {'triangle': {'cat':CatCodes}}   
            )
    ## option 1: make a separate file for each surface
    else: 
        for i in range(nSurfaces):
            filterPoints = CatCodes==UniqueCodes[i]
            nCells = np.sum(filterPoints)
            Cells_i = np.zeros((nCells, 3),dtype ='int')
            cntr = 0
            for j in range(nCells):
                Cells_i[j]=[cntr, cntr+1, cntr+2]
                cntr=cntr+3
  
            meshio.write_points_cells(
                fileprefix+str(i)+".vtk",
                points[np.repeat(filterPoints,3), :],
                cells={'triangle':Cells_i}
                )
    
    return nSurfaces, points, CatCodes

def CalculatePlotStructure(modelfile, plot, includeGravityCalc=0, cubesize = 250,  
                           xy_origin=[317883,4379246, 1200-4000], plotwells =1,
                           outputOption = 1, outputfolder = '', num=0):
    
    output_name = 'ScratchPlots/'
    if(includeGravityCalc==0):
        outputoption = 'BLOCK_SURFACES'
    else:
        outputoption = 'ALL'

    #Calculate the model
    start = time.time()
    sim.calculate_model(modelfile, output_name, outputoption)
    end = time.time()
    print('Calculation time took '+str(end - start) + ' seconds')

    ## Now need to change the DXF file (mesh format) to VTK. 
    ## This is slow unfortunately and I'm sure can be optimized
    start = time.time()
    points, cell_data, faceCounter = getDXF_parsed_structure(output_name)
    end = time.time()
    print('Parsing time took '+str(end - start) + ' seconds')


    ## Make a vtk file for each surface (option 1) 
    # or make a single vtk file for all surfaces (option 2)
    fileprefix = outputfolder+'Surface'
    start = time.time()
    nSurfaces, points, CatCodes = convertSurfaces2VTK(points, cell_data,
                                                      faceCounter, outputOption,
                                                      fileprefix, num=num, 
                                                      xy_origin=xy_origin)   
    end = time.time()
    print('Convert 2 VTK time took '+str(end - start) + ' seconds')

#     ## Now get the lithology data
#     N1 = pynoddy.output.NoddyOutput(output_name)
#     Lithology = N1.block
# #    Lithology=np.swapaxes(Lithology,0,2)


#     lithology = N1.block

#     [maxX, maxY, maxZ] = np.max(points, axis=0)
#     [minX, minY, minZ] = np.min(points, axis=0)
#     minZ = xy_origin[2]
#     x = np.linspace(minX, maxX, N1.nx, dtype=np.float32)
#     y = np.linspace(minY, maxY, N1.ny, dtype=np.float32)
#     z = np.linspace(xy_origin[2], maxZ, N1.nz, dtype=np.float32)
# #    z = np.linspace(0, 4000, N1.nz, dtype=np.float32)

#     delx = x[1]-x[0]
#     dely = y[1]-y[0]
#     delz = z[1]-z[0]
    
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

#     CoordXYZ = np.concatenate((xx.reshape(-1,1),yy.reshape(-1,1),zz.reshape(-1,1)), axis=1)

#     Lithology2 = griddata(CoordXYZ, np.transpose(lithology, axes =(2, 1, 0)).reshape(-1,), (xx, yy, zz), method='nearest')


#     vol = vtkP.Volume(Lithology2, c='jet', spacing=[N1.delx, N1.dely,N1.delz], origin =[xy_origin[0]+N1.xmin, xy_origin[1]+N1.ymin, xy_origin[2]+N1.zmin])
#     lego = vol.legosurface(-1, np.max(Lithology)*2).opacity(0.95).c('jet')
#     plot += lego

    colors = pl.cm.jet(np.linspace(0,1,nSurfaces))

    for i in range(nSurfaces):
        filename = fileprefix+str(i)+'.vtk'
        e=vtkP.load(filename).c(colors[i, 0:3])
    
        plot += e

    return points

num = 503
G = 255

folder = 'C:/Users/ahino/Documents/FinalGeothermicsToday/Top50/Thread'+str(num)+'/HistoryFileInspection/'

picklefiles = glob(folder+'His_'+str(num)+'_G_'+str(G)+'*.his')

# Model with five faults three layers 
modelfile = picklefiles[0]
#modelfile = 'ToyHistory1.his'
#Alter the mesh size if desiring to speed up the process. Recommended size is 100
cubesize = 150
includeGravityCalc = 0
xy_origin=[316448, 4379166, -2700]
xy_extent = [8850, 9035,3900]
vtkP.settings.embedWindow(False) #you can also choose to change to itkwidgets, k3d, False (popup)

plot = vtkP.Plotter(axes=1, bg='white', interactive=1)

points = CalculatePlotStructure(modelfile, plot, includeGravityCalc, cubesize = cubesize, xy_origin=xy_origin, num=num)
plot.show(viewup='z')
vtkP.exportWindow('embryo.x3d')
