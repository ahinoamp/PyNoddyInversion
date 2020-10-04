# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:07:53 2020

@author: ahinoamp
"""

import evd_fast as evd
import signed_distance_functions as sdf
import plt_pos_pri_comp as pltpripos
import numpy as np

def getPC(facies_ndarray_file):
    fac_nums = [0]
    norm_thresh = 2
    i_dim = 74
    j_dim = 76
    k_dim = 26
    #1. signed distance
    print('im here')
    fac_rbnsd_all = sdf.fac_samples_rbnsd(facies_ndarray_file, fac_nums, norm_thresh, i_dim, j_dim, k_dim)
    np.savetxt('fac_rbnsd_all.csv', fac_rbnsd_all, delimiter=',', fmt='%10.5f') 

    #2. pca    
    print('im here')
    m_pri=fac_rbnsd_all
    m_mean_pri = m_pri.mean(axis=0)
    m_eigvec_pri = evd.evd_fast(m_pri, len(m_pri))
    m_pcscr_pri=(m_pri-m_mean_pri).dot(m_eigvec_pri)
    np.savetxt('m_pcscr_pri.csv', m_pcscr_pri, delimiter=',', fmt='%10.5f') 
    return m_pcscr_pri

facies_prior_file = 'Z:/OptimisationPatua/subsPrior300.csv'
facies_posterior_file = 'Z:/OptimisationPatua/newPostT.csv'

facies_arrayPrior = np.loadtxt(facies_prior_file, delimiter=',')
samples_sizePrior, n_features = facies_arrayPrior.shape

facies_arrayPosterior = np.loadtxt(facies_posterior_file, delimiter=',')
samples_sizePosterior, n_features = facies_arrayPosterior.shape

Big = np.concatenate((facies_arrayPrior, facies_arrayPosterior))    
bothfiles = 'Z:/OptimisationPatua/Combo300.csv'
np.savetxt(bothfiles, Big, delimiter=',', fmt='%d') 

Both = getPC(bothfiles)
PC_Prior = Both[:samples_sizePrior, :]
PC_Posterior = Both[samples_sizePrior:, :]

#3. plot
x_compnum = 1
y_compnum = 2
#pltpripos.plt_pos_pri_compNoe(x_compnum, y_compnum, PC_Posterior)
pltpripos.plt_pos_pri_comp(x_compnum, y_compnum, PC_Prior, PC_Posterior)
