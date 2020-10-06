# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 22:25:04 2020

@author: ahinoamp

Compare the results of the four optimisation methods
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.close('all')

file = 'Z:/FinalThesisRun/PlotsCombo/ParametersWithMismatch.csv'

Data = pd.read_csv(file)
Data['Look'] = Data['OptimMethod']

fig, ax = plt.subplots(1, 1, figsize=(8,4))

figOptim = sns.catplot(x="Look", y="MinMismatch", kind="box", data=Data);

#figOptim  = sns.catplot(x="Look", y="MinMismatch", kind="swarm",
#            data=Data.query("MinMismatch <= 0.54"));


figOptim.set_axis_labels("Search method", "Final mismatch")

figOptim.savefig('Compare4.png', dpi = 300,bbox_inches="tight")