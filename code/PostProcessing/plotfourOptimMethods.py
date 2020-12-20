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

file = 'C:/Users/ahino/Documents/GitHub/PyNoddyInversion/code/Combo_Scratch/ParametersWithMismatch.csv'

Data = pd.read_csv(file)

figOptim = sns.catplot(x="OptimMethod", y="MinMismatch", kind="box", data=Data);

figOptim.set_axis_labels("Search method", "Final mismatch")

figOptim.savefig('Compare4.png', dpi = 300,bbox_inches="tight")