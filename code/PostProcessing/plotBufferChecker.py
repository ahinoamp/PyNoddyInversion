# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:24:38 2020

@author: ahinoamp

A visualization of how the buffer checker works, when checking whether two
faults overlap.

"""
from shapely.geometry import Polygon
from shapely.geometry import LineString
import matplotlib.pyplot as plt

def calculate_iou(poly_1, poly_2):
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def calculate_iou1(poly_1, poly_2):
    iou = poly_1.intersection(poly_2).area / poly_1.area
    return iou


line_1 = [[0, 1], [0, 2], [1, 2], [1, 4]]
line_2 = [[0.1, 1], [0.1, 2], [5, 2], [6, 4]]

line1Obj = LineString(line_1)
line2Obj = LineString(line_2)

polygon1obj = line1Obj.buffer(0.1)
polygon2obj = line2Obj.buffer(0.1)

print(calculate_iou(polygon1obj, polygon2obj))
print(calculate_iou1(polygon1obj, polygon2obj))
print(calculate_iou1(polygon2obj, polygon1obj))


x,y = polygon1obj.exterior.xy
plt.plot(x,y)
x,y = polygon2obj.exterior.xy
plt.plot(x,y)
