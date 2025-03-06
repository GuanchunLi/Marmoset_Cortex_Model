# -*- coding: utf-8 -*-
"""
This code provides necessary functions for the analysis of network connectivity and other properties
Created on Thu Feb 27 12:54:04 2020

@author: songt
"""

from __future__ import division
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import random

from numpy import linspace
import statsmodels.tsa.api as smt
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

#local the connectome data of macaque or marmoset network
def load_data(datafile):
    
    plt.close('all')
    
    with open(datafile,'rb') as f:
        p = pickle.load(f, encoding='latin1')   
        
    print('Initializing Model. From ' + datafile + ' load:')
    
    print(p.keys())
    
    p['hier_vals'] = p['hier_vals']/max(p['hier_vals'])    
    p['n_area'] = len(p['areas'])
      
    return p

#set the network parameters and generate the connectiivty matrix W in the linear dynamical system 
def genetate_net_connectivity(p_t,MACAQUE_CASE=0,LINEAR_HIER=0,ZERO_HIER=0,FIT_HIER=0,IDENTICAL_HIER=0,LOCAL_IDENTICAL_HIERARCHY=0,LOCAL_LINEAR_HIERARCHY=0,LONG_RANGE_IDENTICAL_HIERARCHY=0, BREAK_EI=0, SHUFFLE_FLN=0,SHUFFLE_TYPE=0,ZERO_FLN=0,IDENTICAL_FLN=0,STRONG_GBA=0, DELETE_STRONG_LOOP=0,DELETE_CON_DIRECTION=0,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=0,eta_val=0.68,eta_local_val=0.68,long_amp=1,FF_FLN=0, FB_FLN=0, DELETE_WEAK_FLN=0, str_factor=1, diverse_hi=0, enhance_hi=0):
    
    p=p_t.copy()
    
    if LINEAR_HIER+ZERO_HIER+IDENTICAL_HIER+LOCAL_IDENTICAL_HIERARCHY+LOCAL_LINEAR_HIERARCHY+LONG_RANGE_IDENTICAL_HIERARCHY>1 or SHUFFLE_FLN+IDENTICAL_FLN+ZERO_FLN>1:
        raise SystemExit('Conflict of network parameter setting!')
    
    if FIT_HIER:
        p['hier_vals'] = np.load('hier_vals_fit.npy')
        p['hier_vals_inh'] = np.load('hier_vals_fit.npy')

    #scale the hierarchy value linearly    
    if LINEAR_HIER:
        hier_vals = p['hier_vals']
        hier_order = np.argsort(hier_vals)
        hier_linear = np.linspace(0,1,p['n_area'])
        hier_inv_order = np.argsort(hier_order)
        p['hier_vals'] = hier_linear[hier_inv_order]
        print('LINEAR_HIER \n')
        
    #set the hierarchy value to be identical to zero 
    if ZERO_HIER:
        p['hier_vals']=np.zeros(len(p['areas']))
        print('ZERO_HIER \n')
        
    #set the hierarchy value to be identical to its mean
    if IDENTICAL_HIER:
        p['hier_vals']=np.ones(len(p['areas']))*np.mean(p['hier_vals'])
        p['hier_vals_inh']=np.ones(len(p['areas']))*np.mean(p['hier_vals'])
        print('IDENTICAL_HIER \n')
        
    #set the FLN value to be random
    if SHUFFLE_FLN:
        p['fln_mat']=matrix_random_permutation(p,p['fln_mat'],SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE) 
        print('SHUFFLE_FLN \n')
        
    #set the FLN value to be identical to its mean
    if IDENTICAL_FLN:
        #p['fln_mat']=np.ones_like(p['fln_mat'])*np.mean(p['fln_mat'])   #all to all connected
        p['fln_mat'][p['fln_mat']>0]=np.mean(p['fln_mat'][p['fln_mat']>0])  #the topology remains the same but the weight is changed
        print('IDENTICAL_FLN \n')
        
    # disconnect all the inter-area connections 
    if ZERO_FLN:  
        p['fln_mat']=np.zeros_like(p['fln_mat'])
        print('ZERO_FLN \n')
    
    # keep only feed-forward connections
    if FF_FLN:
        p['fln_mat'] = np.tril(p['fln_mat'])
        print('Feedforward FLN only \n')


    # keep only feed-back connections
    if FB_FLN:
        p['fln_mat'] = np.triu(p['fln_mat'])
        print('Feedforward FLN only \n')

    #delete strong loops defined as p['sln_mat'][i,j]>0.575 and p['sln_mat'][j,i]>0.575:
    if DELETE_STRONG_LOOP:
        #---------------------------------------------------------------------------------
        # identify the strong loops
        #--------------------------------------------------------------------------------- 
        sl_target_list=[]
        sl_source_list=[]
        
        for i in np.arange(p['n_area']):
            for j in np.arange(i+1,p['n_area']):
                if p['sln_mat'][i,j]>0.575 and p['sln_mat'][j,i]>0.575:
                    sl_target_list.append(i)
                    sl_source_list.append(j)
                    print(p['areas'][j]+' and '+p['areas'][i])
        print('strong loop number=',len(sl_target_list))
        
        for i, j in zip(sl_target_list,sl_source_list): 
            p['fln_mat'][i,j]=0
            p['fln_mat'][j,i]=0
        print('DELETE_STRONG_LOOP \n')
        
    #delete the direction of connections by making the connectivity matrix symmetric    
    if DELETE_CON_DIRECTION:
        p['fln_mat']=(p['fln_mat']+p['fln_mat'].T)/2
        print('DELETE_CON_DIRECTION \n')
    
    if DELETE_WEAK_FLN:
        fln_sorted = np.sort(p['fln_mat'][p['fln_mat'] > 0])
        keep_ratio = 0.5
        fln_threshold = fln_sorted[int(len(fln_sorted)*(1-keep_ratio))]
        p['fln_mat'] = p['fln_mat'] * (p['fln_mat'] > fln_threshold)
        
#---------------------------------------------------------------------------------
# Network Parameters
#---------------------------------------------------------------------------------
    p['beta_exc'] = 0.066  # Hz/pA
    p['beta_inh'] = 0.351  # Hz/pA
    p['tau_exc'] = 20  # ms
    p['tau_inh'] = 10  # ms
    # p['wEE'] = 24.4 * str_factor # pA/Hz
    p['wEE'] = 24.4 * str_factor # pA/Hz
    p['wIE'] = 12.2 * str_factor # pA/Hz
    p['wEI'] = 19.7 * str_factor # pA/Hz
    # p['wII'] = 12.5 * str_factor # pA/Hz 
    p['wII'] = 12.5 * str_factor
    p['muEE']= 33.7 * long_amp * str_factor# pA/Hz  33.7#TEST TEST TEST 
    p['muIE'] = 25.3 * long_amp * str_factor # pA/Hz  25.3  or smaller delta set 25.5
    p['eta_local'] = eta_local_val # 0.68 original
    p['eta'] = eta_val
    p['eta_inh_local'] = eta_local_val # 0.68 original
    p['eta_inh'] = eta_val
    p['hier_vals_inh'] = p['hier_vals']
    # p['wEE'] = 2.44 * str_factor
    # p['wIE'] = 1.22 * str_factor
    # p['beta_exc'] = 0.66
    
    if CONSENSUS_CASE:
        if MACAQUE_CASE:
            p['muEE']=33.3   # pA/Hz  33.7
            print('CONSENSUS_CASE=1, pay attention! Now muEE=',p['muEE'])
        else:
            p['muEE']=33.7   # pA/Hz  33.7
            print('CONSENSUS_CASE=1, pay attention! Now muEE=',p['muEE'])
    else:
        print('CONSENSUS_CASE=0')    
    
    #when the mini-cost network is reconstructed, the parameter muEE needs to be adjusted to avoid positive eigenvalue
    if SHUFFLE_TYPE==5:
        if MACAQUE_CASE:
             p['muEE'] =  32.2  #33.7   # pA/Hz   #CHANGED!!!!!!!
        else:
             p['muEE'] = 33.2   #33.2   # pA/Hz   #CHANGED!!!!!!!
        print('mini-cost network is reconstructed, pay attention! Now muEE=',p['muEE'])
                
    #strong GBA regime from joglekar etal 2018
    if STRONG_GBA == 1:
        if MACAQUE_CASE:
            p['wEI'] = 25.2 * str_factor # pA/Hz
            p['muEE'] = 51.5 * str_factor # pA/Hz
        else:
            p['wEI'] = 25.2 * str_factor # pA/Hz
            p['muEE'] = 48.5 * str_factor # pA/Hz
        print('STRONG_GBA \n')
    elif STRONG_GBA == 2:
        if MACAQUE_CASE:
            p['wEI'] = 25.2 * str_factor # 12.2 pA/Hz
            p['wII'] = 16.8 * str_factor # 24.4 pA/Hz
        else:
            p['wEI'] = 25.2  # pA/Hz
            p['muEE'] = 48.5  # pA/Hz
            p['wIE'] = 11.3  # pA/Hz
            p['wII'] = 13.2  # pA/Hz
            p['wEE'] = 26.2 # pA/Hz
            p['muIE'] = 29.7 # pA/Hz  25.3  or smaller delta set 25.5
        print('STRONG_GBA:2 \n')
    elif STRONG_GBA == 3:
        p['wEI'] = 23.6 * str_factor
        p['muEE'] = 48.5 * str_factor # pA/Hz
        print('STRONG_GBA:3 \n')
    else:
        print('No STRONG GBA')
    
    if diverse_hi == 1:
        p['beta_inh'] /= 1
        # p['beta_exc'] *= 1
        p['beta_exc'] = p['beta_inh'] / 2
        p['wII'] *= 1.5 # 1.4
        p['wIE'] *= 1.
        p['wEI'] *= 1.
        p['muIE'] *= 1. #1/08
        p['wEE'] = 14.785915973473635 # Good: 15.504291289766263  
        p['eta_local'] = 0.6436289746755808 # Good: 0.6193180940525973 
        p['muEE'] = 23.700851481795087*1.02 # Good: 25.198690673388548 * 1.1
        p['eta_inh'] = 0.0954912785774767 # Good: 0
        p['eta_inh_local'] = 0.68
        p['eta'] *= 1e-1 # 0
        
    elif diverse_hi == 2:
        p['beta_inh'] /= 1
        # p['beta_exc'] *= 1.5
        p['beta_exc'] = p['beta_inh'] / 3
        p['wII'] *= 2.5 # 1.6
        p['wIE'] *= 2.8
        p['wEI'] *= 2
        p['muIE'] *= 1.3
        p['wEE'] = 45.211352728098  # 12.467895286394006
        p['eta_local'] = 0.6531639977965071 # 0.6416378458007311
        p['muEE'] = 38.48536994182971 # 22.371088235294117
        p['eta_inh'] = 0.68 # 0.7382939759036146 # 0.7125913043478262
        p['eta_inh_local'] = 0.68
        p['eta'] *= 1

    elif diverse_hi == 3:
        p['beta_inh'] *= 1
        # p['beta_exc'] *= 1.5
        p['beta_exc'] = p['beta_inh'] / 3
        p['wII'] *= 1 # 1.6
        p['muEE'] *= 1.8
        p['wIE'] = 14.556842174092381
        p['eta_inh_local'] = 0.7425045617327324
        p['muIE'] = 45.00535897435898 # 44.984926780626786
        p['eta_inh'] = 0.75664875 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
        p['eta'] += 1e-3
        # p['hier_vals_inh'] = np.load('hier_vals_inh.npy')

    elif diverse_hi == 4:
        p['beta_inh'] *= 1
        p['beta_exc'] *= 1
        p['wII'] *= 1 # 1.6
        p['muEE'] *= 2 # 2
        p['wIE'] = 11.660380103696516  # 12.2
        if MACAQUE_CASE:
            p['eta_inh_local'] = 0.7436837102669386
            p['muIE'] = 45.18349373219373 # 25.3
            p['eta_inh'] = 0.74677596 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
            # p['eta'] += 7e-3
        else:
            p['eta_inh_local'] = 0.7436837102669386
            p['muIE'] = 49.81041566951567 # 25.3
            p['eta_inh'] = 0.76508359 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
            p['eta'] += 7e-3


    elif diverse_hi == 41:
        p['beta_inh'] *= 1
        p['beta_exc'] *= 1
        p['wII'] *= 1 # 1.6
        p['muEE'] *= 2 # 2
        p['wIE'] = 11.660380103696516  # 12.2
        if MACAQUE_CASE:
            p['eta_inh_local'] = 0.745
            p['muIE'] = 45.18349373219373 # 25.3
            p['eta_inh'] = 0.745 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
            # p['eta'] += 7e-3
        else:
            p['eta_inh_local'] = 0.76 # 0.7436837102669386
            p['muIE'] = 49.81041566951567 # 25.3
            p['eta_inh'] = 0.76 # 0.76508359 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
            p['eta'] += 5e-3
            p['eta_local'] += 5e-3

        # p['hier_vals_inh'] = np.load('hier_vals_inh.npy')
    elif diverse_hi == 5:
        # FIT STRONG GBA
        # p['eta_inh_local'] = 0.7436837102669386 * 0.8
        # p['muIE'] *= 1.08
        # p['muEE'] *= 1.1
        # p['wEI'] *= 1.21
        p['wII'] *= 1.01

    elif diverse_hi == 6:
        p['beta_inh'] *= 1
        p['beta_exc'] *= 1
        p['wII'] *= 1 # 1.6
        p['muEE'] *= 3 # 2
        p['wIE'] = 11.660380103696516  # 12.2
        p['eta_inh_local'] = 0.7436837102669386 + 3e-3
        p['muIE'] = 75.00412258425209 # 25.3
        p['eta_inh'] = 0.7563872489049069 # 0.75758736 + 1e-4 # 0.7382939759036146 # 0.7125913043478262
        p['eta'] += 0e-3

    if BREAK_EI == 1:
        # p['wEI'] *= 1.02 # pA/Hz
        p['muEE'] *= 0.99 # pA/Hz

    if enhance_hi == 1:
        p['eta_local'] *= 1.
        # lamb_1 = - 0.02
        # lamb_2 = - 0.001

        # a0 = p['beta_exc'] / p['tau_exc'] * (p['wEE'] - 1/p['beta_exc'])
        # a1 = p['beta_exc'] / p['tau_exc'] * p['wEE'] * p['eta']
        # d = p['beta_inh'] / p['tau_inh'] * (p['wII'] + 1/p['beta_inh']) #0.53875; 
        # b = p['beta_exc'] / p['tau_exc'] * p['wEI']
        # c0 = (d*a0 + (a0-d)*lamb_1) / b
        # c1 = (d*a1 + a1*lamb_2 + (a0-d)*(lamb_2 - lamb_1)) / b
        p['eta_inh_local'] *= 1

        p['muEE'] *= 1/2
        p['muIE'] *= 1/2

    p['exc_scale'] = (1+p['eta']*p['hier_vals'])
    p['local_exc_scale'] = (1+p['eta_local']*p['hier_vals'])
    p['inh_scale'] = (1+p['eta_inh']*p['hier_vals_inh'])
    p['local_inh_scale'] = (1+p['eta_inh_local']*p['hier_vals'])
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
    
    fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
    fln_scaled_inh = (p['inh_scale'] * p['fln_mat'].T).T
    
    #Modeling the case that feedforward connection prefers to E population, and feedback connection prefers to I population
    if LONGRANGE_EI_ASYMMETRY==1:
        fln_scaled_EE=(p['exc_scale'] * p['fln_mat'].T * p['sln_mat'].T).T
        fln_scaled_IE=(p['exc_scale'] * p['fln_mat'].T * (1-p['sln_mat']).T).T
        print('LONGRANGE_EI_ASYMMETRY \n')
        
    #keep local hierarchy gradient, and set the long-range connection independent of hierarchy
    if LONG_RANGE_IDENTICAL_HIERARCHY:
        long_range_hier_vals=np.ones(len(p['areas']))*np.mean(p['hier_vals'])
        long_range_exc_scale=(1+p['eta']*long_range_hier_vals)
        fln_scaled = (long_range_exc_scale * p['fln_mat'].T).T
        print('LONG_RANGE_IDENTICAL_HIERARCHY \n')
        
    #keep long-range hierarchy graident, and set the local connection independent of hierarchy
    if LOCAL_IDENTICAL_HIERARCHY:
        local_hier_vals=np.ones(len(p['areas']))*np.mean(p['hier_vals'])
        # p['exc_scale'] = (1+p['eta']*local_hier_vals)
        p['local_exc_scale'] = (1+local_hier_vals)
        p['local_inh_scale'] = (1+local_hier_vals)
        local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
        local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        print('LOCAL_IDENTICAL_HIERARCHY \n')
        
    #keep long-range hierarchy graident, and set the local connection dependent of linear hierarchy
    if LOCAL_LINEAR_HIERARCHY:
        local_hier_vals=np.linspace(0,1,p['n_area'])
        p['exc_scale'] = (1+p['eta']*local_hier_vals)
        local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
        local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        print('LOCAL_LINEAR_HIERARCHY \n')
        
    #change the connection weight between a few of areas
    if GATING_PATHWAY:
        VISUAL_INPUT=1
        if MACAQUE_CASE:
            if VISUAL_INPUT:
                area_name_list = ['V4','8m']
            else:
                area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
        else:
            if VISUAL_INPUT:
                area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
            else:
                area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
             
        for name in area_name_list:
            area_idx=p['areas'].index(name)
            local_EI[area_idx]=local_EI[area_idx]*0.9
        print('GATING_PATHWAY \n')
        
    #---------------------------------------------------------------------------------
    # compute the connectivity matrix
    #---------------------------------------------------------------------------------
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):

        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

        if LONGRANGE_EI_ASYMMETRY==1:
             W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled_EE[i,:]/p['tau_exc']
             W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled_IE[i,:]/p['tau_inh']
        else:
            W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
            W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled[i,:]/p['tau_inh']
                
    return p, W

# plot network properties including hierarchy and connectivity matrix    
def plot_network_property(p_t,W_t,dist_mat_t):
    p=p_t.copy()
    W=W_t.copy()
    
    #---------------------------------------------------------------------------------
    # plot FLN matrix
    #---------------------------------------------------------------------------------
    inferno_r = cm.get_cmap('inferno_r', 1024)
    newcolors = inferno_r(np.linspace(0, 1, 1024))
    black = np.array([1, 1, 1, 1]) #[R,G,B,alpha] ranging from 0 to 1
    newcolors[:1, :] = black       # set first color to black
    newcmp = ListedColormap(newcolors)
    
    p['fln_mat'][p['fln_mat']==0]=1e-10
    
    fig, ax = plt.subplots(1,2,figsize=(30,10))
    f=ax[0].pcolormesh(p['fln_mat'],cmap=newcmp,norm=LogNorm(vmin=1e-7, vmax=1))
    
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xticks(x[::2])
    ax[0].set_xticklabels(xticklabels_even)
    ax[0].set_yticks(y[::2])
    ax[0].set_yticklabels(yticklabels_even)
    ax[0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax[0].twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    ax[0].set_title('FLN Matrix')
    ax[0].set_ylabel('Target')
    ax[0].set_xlabel('Source')
    fig.colorbar(f,ax=ax[0])
    
    f=ax[1].pcolormesh(p['fln_mat'],cmap='hot')
    
    # set original ticks and ticklabels
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_xticks(x[::2])
    ax[1].set_xticklabels(xticklabels_even)
    ax[1].set_yticks(y[::2])
    ax[1].set_yticklabels(yticklabels_even)
    ax[1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    
    # second x axis
    ax4 = ax[1].twiny()
    ax4.set_xlim(xlim)
    ax4.set_xticks(x[1::2])
    ax4.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax4.get_xticklabels(), rotation=90)
    
    # second y axis
    ax5 = ax[1].twinx()
    ax5.set_ylim(ylim)
    ax5.set_yticks(y[1::2])
    ax5.set_yticklabels(yticklabels_odd)
    ax5.invert_yaxis()   
    
    ax[1].set_title('FLN Matrix')
    ax[1].set_ylabel('Target')
    ax[1].set_xlabel('Source')
    
    fig.colorbar(f,ax=ax[1])
    
    symmetric_fln=p['fln_mat']+p['fln_mat'].T
    np.fill_diagonal(symmetric_fln,np.max(symmetric_fln))
    fig, ax = plt.subplots(1,2,figsize=(30,10))
    f=ax[0].pcolormesh(symmetric_fln,cmap=newcmp,norm=LogNorm(vmin=1e-7, vmax=1))
    
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xticks(x[::2])
    ax[0].set_xticklabels(xticklabels_even)
    ax[0].set_yticks(y[::2])
    ax[0].set_yticklabels(yticklabels_even)
    ax[0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax[0].twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    ax[0].set_title('FLN Matrix')
    ax[0].set_ylabel('Target')
    ax[0].set_xlabel('Source')
    fig.colorbar(f,ax=ax[0])
    
    f=ax[1].pcolormesh(symmetric_fln,cmap='hot')
    
    # set original ticks and ticklabels
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_xticks(x[::2])
    ax[1].set_xticklabels(xticklabels_even)
    ax[1].set_yticks(y[::2])
    ax[1].set_yticklabels(yticklabels_even)
    ax[1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    
    # second x axis
    ax4 = ax[1].twiny()
    ax4.set_xlim(xlim)
    ax4.set_xticks(x[1::2])
    ax4.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax4.get_xticklabels(), rotation=90)
    
    # second y axis
    ax5 = ax[1].twinx()
    ax5.set_ylim(ylim)
    ax5.set_yticks(y[1::2])
    ax5.set_yticklabels(yticklabels_odd)
    ax5.invert_yaxis()   
    
    ax[1].set_title('symmetric FLN Matrix')
    ax[1].set_ylabel('Target')
    ax[1].set_xlabel('Source')
    
    fig.colorbar(f,ax=ax[1])
    
    #---------------------------------------------------------------------------------
    # plot SLN matrix
    #---------------------------------------------------------------------------------
    bwr = cm.get_cmap('bwr', 1024)
    newcolors = bwr(np.linspace(0, 1, 1024))
    white = np.array([0,0,0, 0.3]) #[R,G,B,alpha] ranging from 0 to 1
    newcolors[:1, :] = white       # set first color to black
    newcmp = ListedColormap(newcolors)
    
    p['sln_mat'][p['sln_mat']==0]=1e-10
    
    fig, ax = plt.subplots(figsize=(13,10))
    f=ax.pcolormesh(p['sln_mat'],cmap=newcmp)
    
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(xticklabels_even)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    ax.set_title('SLN Matrix')
    ax.set_ylabel('Target')
    ax.set_xlabel('Source')
    fig.colorbar(f,ax=ax)

    #---------------------------------------------------------------------------------
    # plot hierachary curve
    #---------------------------------------------------------------------------------
    fig,ax=plt.subplots()
    ax.scatter(range(p['n_area']),p['hier_vals'],30)
    ax.set_xlabel('area')
    ax.set_ylabel('hierarchy')
    ax.set_xticks(range(p['n_area']))
    ax.set_xticklabels(p['areas'])
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.savefig('result/hierarchy.pdf')    
    
    #---------------------------------------------------------------------------------
    # plot connectivity matrix
    #---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1,2,figsize=(30,10))
    f=ax[0].pcolormesh(W,vmin = -1, vmax = 1, cmap='bwr')
    fig.colorbar(f,ax=ax[0],pad=0.15)
    ax[0].set_title('connectivity matrix') 
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xticks(x[::4])
    ax[0].set_xticklabels(xticklabels_even)
    ax[0].set_yticks(y[::4])
    ax[0].set_yticklabels(yticklabels_even)
    ax[0].invert_yaxis()
    ax[0].set_ylabel('Target')
    ax[0].set_xlabel('Source')
    # rotate xticklabels to 90 degree
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax[0].twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[2::4])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[2::4])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()    
    #---------------------------------------------------------------------------------
    #reshape the connectivity matrix by E and I population blocks, get EE block
    #---------------------------------------------------------------------------------
    F_EE=W.copy()[0::2,0::2]
    np.fill_diagonal(F_EE,0)  
    
    f=ax[1].pcolormesh(F_EE, cmap='hot',norm=LogNorm(vmin=1e-3, vmax=np.max(F_EE)))
    fig.colorbar(f,ax=ax[1],pad=0.15)
    ax[1].set_title('EE connectivity matrix') 
    ax[1].set_ylabel('Target')
    ax[1].set_xlabel('Source')
    
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_xticks(x[::2])
    ax[1].set_xticklabels(xticklabels_even)
    ax[1].set_yticks(y[::2])
    ax[1].set_yticklabels(yticklabels_even)
    ax[1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    
    # second x axis
    ax4 = ax[1].twiny()
    ax4.set_xlim(xlim)
    ax4.set_xticks(x[1::2])
    ax4.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax4.get_xticklabels(), rotation=90)
    
    # second y axis
    ax5 = ax[1].twinx()
    ax5.set_ylim(ylim)
    ax5.set_yticks(y[1::2])
    ax5.set_yticklabels(yticklabels_odd)
    ax5.invert_yaxis()   
     
    fig.savefig('result/Connectivity_Matrix.pdf')   
    #---------------------------------------------------------------------------------
    # plot hierachary difference vs FLN
    #---------------------------------------------------------------------------------
    
    fig,ax=plt.subplots(1,3,figsize=(30,10))
    sum_fln=np.sum(p['fln_mat'],axis=1)
    ax[0].scatter(p['hier_vals'],sum_fln,30)
    ax[0].set_xlabel('hierarchy')
    ax[0].set_ylabel('summed FLN for each node')
    
    hier_diff_mat=np.zeros_like(F_EE)     #F_EE_ij is proportional to FLN_ij(1+eta h_i)
    per_mat=np.zeros_like(F_EE) 
    le=len(p['hier_vals'])
    for i in np.arange(le):
        for j in np.arange(le):
            factor=np.abs(p['hier_vals'][i]-p['hier_vals'][j])
            if factor>0:
                hier_diff_mat[i,j]=factor
                per_mat[i,j]=F_EE[i,j]/factor
            
    ax[1].scatter(hier_diff_mat[F_EE>0],F_EE[F_EE>0],30)
    ax[1].set_xlabel('hierarchy difference')
    ax[1].set_ylabel('F_EE')        
    
    ax[2].hist(per_mat[per_mat>0],bins=100)
    ax[2].set_xlabel('effective perturabation elements')
    ax[2].set_yscale('log')
        
    if np.all(dist_mat_t.flatten()==0):
        return;
    else:
        dist_mat=dist_mat_t.copy()
        dist_mat=dist_mat[:p['n_area'],:p['n_area']]
        #---------------------------------------------------------------------------------
        # plot distance matrix
        #---------------------------------------------------------------------------------
        #hot = cm.get_cmap('hot', 1024)
       
        fig, ax = plt.subplots(figsize=(13,10))
        f=ax.pcolormesh(dist_mat,cmap='hot')
        
        x = np.arange(p['n_area']) # xticks
        y = np.arange(p['n_area']) # yticks
        xlim = (0,p['n_area'])
        ylim = (0,p['n_area'])
        
        xticklabels_odd  = p['areas'][1::2]
        xticklabels_even = p['areas'][::2]
        yticklabels_odd=xticklabels_odd
        yticklabels_even=xticklabels_even
        
        # set original ticks and ticklabels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(xticklabels_even)
        ax.set_yticks(y[::2])
        ax.set_yticklabels(yticklabels_even)
        ax.invert_yaxis()
        ax.set_ylabel('Target')
        ax.set_xlabel('Source')
        # rotate xticklabels to 90 degree
        plt.setp(ax.get_xticklabels(), rotation=90)
        
        # second x axis
        ax2 = ax.twiny()
        ax2.set_xlim(xlim)
        ax2.set_xticks(x[1::2])
        ax2.set_xticklabels(xticklabels_odd)
        # rotate xticklabels to 90 degree
        plt.setp(ax2.get_xticklabels(), rotation=90)
        
        # second y axis
        ax3 = ax.twinx()
        ax3.set_ylim(ylim)
        ax3.set_yticks(y[1::2])
        ax3.set_yticklabels(yticklabels_odd)
        ax3.invert_yaxis()   
        
        ax.set_title('Distance Matrix')
        fig.colorbar(f,ax=ax)
        
        
        #---------------------------------------------------------------------------------
        # plot FLN versus distance
        #---------------------------------------------------------------------------------
        fln_flatten=p['fln_mat'].flatten()
        dist_flatten=dist_mat.flatten()
        fln_flatten_nonzero=fln_flatten[fln_flatten>1e-9]
        dist_flatten_nonzero=dist_flatten[fln_flatten>1e-9]
        fig,ax=plt.subplots()
        ax.scatter(dist_flatten_nonzero,np.log(fln_flatten_nonzero),10)
        ax.set_xlabel('distance')
        ax.set_ylabel('FLN')
        plt.setp(ax.get_xticklabels(), rotation=90)
        
        #---------------------------------------------------------------------------------
        # plot FLN times distance
        #---------------------------------------------------------------------------------
        fln_flatten=p['fln_mat'].flatten()
        dist_flatten=dist_mat.flatten()
        fln_flatten_nonzero=fln_flatten[fln_flatten>1e-9]
        dist_flatten_nonzero=dist_flatten[fln_flatten>1e-9]
        fln_times_dist=fln_flatten_nonzero*dist_flatten_nonzero
        fig,ax=plt.subplots()
        ax.hist(fln_times_dist,rwidth=0.8)
        ax.set_xlabel('FLN*distance')
        ax.set_ylabel('counts')
        ax.set_title(['cost=',str(np.sum(fln_times_dist))])
        plt.setp(ax.get_xticklabels(), rotation=90)
        #---------------------------------------------------------------------------------
        # plot distance distribution
        #---------------------------------------------------------------------------------
        # fig,ax=plt.subplots(1,2)
        # ax[0].hist(dist_flatten)
        # ax[0].set_xlabel('distance')
        # ax[0].set_ylabel('counts')
        
        # exp_mat=np.zeros_like(dist_mat)
        # for i in np.arange(p['n_area']):
        #     for j in np.arange(p['n_area']): 
        #         exp_mat[i,j]=np.exp(-dist_mat[i,j]/np.mean(dist_mat))
        
        # ax[1].hist(exp_mat.flatten())
        # ax[1].set_xlabel('exp_weights')
        # ax[1].set_ylabel('counts')
        
#eig mode decomposition of the connectivity matrix
def eig_decomposition(p_t,W_t,EI_REARRANGE=1,MACAQUE_CASE=1,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CLOSE_FIG=0):
    p=p_t.copy()
    W=W_t.copy()
    
    if EI_REARRANGE==1:
        W_EI=np.zeros_like(W)
        W_EI[0:p['n_area'],0:p['n_area']]=W.copy()[0::2,0::2]
        W_EI[0:p['n_area'],p['n_area']:]=W.copy()[0::2,1::2]
        W_EI[p['n_area']:,0:p['n_area']]=W.copy()[1::2,0::2]
        W_EI[p['n_area']:,p['n_area']:]=W.copy()[1::2,1::2]
    else:
        W_EI=W
    
    #---------------------------------------------------------------------------------
    # eigenmode decomposition
    #--------------------------------------------------------------------------------- 
    eigVals, eigVecs = np.linalg.eig(W_EI)
    
    normality_meaure(W_EI[0:p['n_area'],0:p['n_area']])
    
    eigVecs_a=np.abs(eigVecs)
    
    tau=-1/np.real(eigVals)
    tau_s=np.zeros_like(tau)
    for i in range(len(tau)):
        tau_s[i]=format(tau[i],'.2f')
    
    ind=np.argsort(-tau_s)
    eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))
    eigVecs_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
    tau_reorder=np.zeros(2*p['n_area'])
    
    for i in range(2*p['n_area']):
        eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
        eigVecs_reorder[:,i]=eigVecs[:,ind[i]]
        tau_reorder[i]=tau_s[ind[i]]
     
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigVecs_a_reorder),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W matrix eigenvector visualization')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(tau_reorder[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')    

    #---------------------------------------------------------------------------------
    # get the slowest eigvalue
    #--------------------------------------------------------------------------------- 
    
    eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
    eigVecs_slow=normalize_matrix(eigVecs_slow,column=1)
    tau_slow=tau_reorder[:p['n_area']]
        
    fig, ax = plt.subplots()
    
    f=ax.pcolormesh(eigVecs_slow,cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.15)
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    x = np.arange(len(tau_slow)) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,len(tau_slow))
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::5])
    ax.invert_xaxis()
       
    ax.set_xticklabels(tau_slow[::5])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    ax.set_title('slow eigenvector visualization')
    fig.savefig('result/slowest_eigenmode.pdf')    
    
    if CLOSE_FIG==1:
        plt.close('all')
        
    #---------------------------------------------------------------------------------
    # run the simulation to check the response at each area
    #---------------------------------------------------------------------------------
    # if LONGRANGE_EI_ASYMMETRY==0:
    #     run_stimulus(p,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=GATING_PATHWAY)
    # else:
    #     run_stimulus_longrange_ei_asymmetry(p,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=MACAQUE_CASE)
        
    #plt_white_noise_input(p,VISUAL_INPUT=1,MACAQUE_CASE=MACAQUE_CASE,NO_gradient=0,NO_feedback=0,NO_long_link=0)

    #theoretical_time_constant_input_at_one_area(p,eigVecs,eigVals,input_loc='V1')
    #theoretical_time_constant_input_at_all_areas(p,eigVecs,eigVals)  
        
    return eigVecs_a_reorder, tau_reorder

def eigen_structure_approximation(p_t,MACAQUE_CASE=1,SHUFFLE_FLN=0,SHUFFLE_TYPE=0,CONSENSUS_CASE=0):

    p=p_t.copy()
    
    _,W0=genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=1,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    p,W1=genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=0,SHUFFLE_FLN=SHUFFLE_FLN,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    
    theta=p['beta_exc']*p['muEE']/p['tau_exc']/(p['beta_inh']*p['muIE']/p['tau_inh'])
    print('theta=',theta)
    
    #---------------------------------------------------------------------------------
    #reshape the connectivity matrix by E and I population blocks, EE, EI, IE, II
    #---------------------------------------------------------------------------------
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    #the variable names are consistent with symbols used in the notes
    D=W0_EI
    F=W1_EI-W0_EI
    
    D_EE=W0_EI[0:p['n_area'],0:p['n_area']]
    D_IE=W0_EI[p['n_area']:,0:p['n_area']]
    D_EI=W0_EI[0:p['n_area'],p['n_area']:]
    D_II=W0_EI[p['n_area']:,p['n_area']:]
    
    F_EE=F[0:p['n_area'],0:p['n_area']]
    F_IE=F[p['n_area']:,0:p['n_area']]
    
    cand_dei=-np.diag(D_EE)/np.diag(D_IE)*np.diag(D_II)*p['tau_exc']/p['beta_exc']
    
    fig,ax = plt.subplots()
    ax.plot(np.arange(p['n_area']),cand_dei,'-o')
    ax.set_title('for choosing wEI candidate')
    ax.set_xlabel('wEI index')
    ax.set_ylabel('wEI value that gives zero eigenvals')
    print('min wEI=',max(cand_dei))
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(D_EE,cmap='bwr')
    ax[0,0].set_title('D_EE')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(D_IE,cmap='bwr')
    ax[0,1].set_title('D_IE')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(F_EE,cmap='bwr')
    ax[0,2].set_title('F_EE')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(D_EI,cmap='bwr')
    ax[1,0].set_title('D_EI')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(D_II,cmap='bwr')
    ax[1,1].set_title('D_II')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(F_IE,cmap='bwr')
    ax[1,2].set_title('F_IE')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #approximations of A and B (see notes for detailed derivations)
    #--------------------------------------------------------------------------
    A=np.zeros_like(D_EE)
    A_app=np.zeros_like(A)
    B=np.zeros_like(A)
    B_app=np.zeros_like(A)
    
    for i in np.arange(p['n_area']):
        A[i,i]=0.5/D_IE[i,i]*(D_II[i,i]-D_EE[i,i]+np.sqrt((D_EE[i,i]+D_II[i,i])**2-4*(D_EE[i,i]*D_II[i,i]-D_EI[i,i]*D_IE[i,i])))
        A_app[i,i]=-D_EI[i,i]/D_II[i,i]
        B[i,i]=-D_IE[i,i]/(D_EE[i,i]+2*D_IE[i,i]*A[i,i]-D_II[i,i])
        B_app[i,i]=D_IE[i,i]/D_II[i,i]
        
    print('mean_A=',np.mean(np.diag(A)))
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(A,cmap='hot_r')
    ax[0,0].set_title('A')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(A_app,cmap='hot_r')
    ax[0,1].set_title('A_app')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(A-A_app,cmap='hot_r')
    ax[0,2].set_title('A-A_app')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(B,cmap='hot_r')
    ax[1,0].set_title('B')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(B_app,cmap='hot_r')
    ax[1,1].set_title('B_app')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(B-B_app,cmap='hot_r')
    ax[1,2].set_title('B-B_app')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #approximations of eigenvalues
    #--------------------------------------------------------------------------
    fig, ax = plt.subplots(1,2)
    ax[0].plot(np.arange(p['n_area']),np.diag(D_EE+A@D_IE),'o',label='real')
    ax[0].plot(np.arange(p['n_area']),np.diag(D_EE+A_app@D_IE),'-',label='approximated')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Eigenvalue')
    ax[0].legend()
    ax[1].plot(np.arange(p['n_area']),np.diag(D_II-A@D_IE),'o',label='real')
    ax[1].plot(np.arange(p['n_area']),np.diag(D_II-A_app@D_IE),'-',label='approximated')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Eigenvalue')
    ax[1].legend()
    
    #--------------------------------------------------------------------------
    #compute P to diagnalize the local connectivity matrix without long-range connectivity
    #--------------------------------------------------------------------------
    P=np.zeros((2*p['n_area'],2*p['n_area']))
    P[0:p['n_area'],0:p['n_area']]=np.eye(p['n_area'])
    P[0:p['n_area'],p['n_area']:]=A
    P[p['n_area']:,0:p['n_area']]=B
    P[p['n_area']:,p['n_area']:]=np.eye(p['n_area'])+A@B
    P_inv=np.linalg.inv(P)
    
    fig, ax = plt.subplots(1,2)
    f=ax[0].pcolormesh(P,cmap='bwr')
    ax[0].set_title('P')
    ax[0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0],pad=0.1)
    f=ax[1].pcolormesh(P_inv,cmap='bwr')
    ax[1].set_title('P_inv')
    ax[1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1],pad=0.1)
    
    #--------------------------------------------------------------------------
    #similarity transform on the connectivity matrix using P
    #--------------------------------------------------------------------------
    Lambda=P@D@P_inv
    Sigma=P@F@P_inv
    Lambda[np.abs(Lambda)<1e-12]=0
    Sigma[np.abs(Sigma)<1e-12]=0
    
    Gamma=Lambda+Sigma
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(Lambda,cmap='bwr')
    ax[0,0].set_title('Lambda')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(Sigma,cmap='bwr')
    ax[0,1].set_title('Sigma')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh(Gamma,cmap='bwr')
    ax[0,2].set_title('Gamma')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    f=ax[1,0].pcolormesh(np.abs(Lambda),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,0].set_title('|Lambda|')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(np.abs(Sigma),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,1].set_title('|Sigma|')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(np.abs(Gamma),cmap='hot_r',norm=LogNorm(vmin=None,vmax=None))
    ax[1,2].set_title('|Gamma|')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    
    #--------------------------------------------------------------------------
    #extract block matrices after similarity transformation on the connectivity matrix
    #--------------------------------------------------------------------------
    Sigma_1=Sigma[0:p['n_area'],0:p['n_area']]
    Sigma_2=Sigma[0:p['n_area'],p['n_area']:]
    Sigma_3=Sigma[p['n_area']:,0:p['n_area']]
    Sigma_4=Sigma[p['n_area']:,p['n_area']:]
    Lambda_1=Lambda[0:p['n_area'],0:p['n_area']]
    Lambda_4=Lambda[p['n_area']:,p['n_area']:]
    
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].pcolormesh(F_EE+A@F_IE,cmap='bwr')
    ax[0,0].set_title('F_EE+A@F_IE')
    ax[0,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    f=ax[0,1].pcolormesh(np.eye(p['n_area'])+A@B,cmap='bwr')
    ax[0,1].set_title('I+A@B')
    ax[0,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    f=ax[0,2].pcolormesh((F_EE+A@F_IE)@(np.eye(p['n_area'])+A@B),cmap='bwr')
    ax[0,2].set_title('Sigma1')
    ax[0,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[0,2],pad=0.1)
    
    f=ax[1,0].pcolormesh(F_EE,cmap='bwr')
    ax[1,0].set_title('F_EE')
    ax[1,0].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    f=ax[1,1].pcolormesh(A,cmap='bwr')
    ax[1,1].set_title('A')
    ax[1,1].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    f=ax[1,2].pcolormesh(A@F_IE,cmap='bwr')
    ax[1,2].set_title('A@F_IE')
    ax[1,2].invert_yaxis()  
    fig.colorbar(f,ax=ax[1,2],pad=0.1)
    
    #--------------------------------------------------------------------------
    #estimate the order of elements in block matrices
    #--------------------------------------------------------------------------
    Sigma_1_flat=Sigma_1.flatten()
    Sigma_1_sort=np.sort(np.abs(Sigma_1_flat))
    Sigma_2_flat=Sigma_2.flatten()
    Sigma_2_sort=np.sort(np.abs(Sigma_2_flat))
    Sigma_3_flat=Sigma_3.flatten()
    Sigma_3_sort=np.sort(np.abs(Sigma_3_flat))
    Sigma_4_flat=Sigma_4.flatten()
    Sigma_4_sort=np.sort(np.abs(Sigma_4_flat))
    
    Lambda_1_flat=Lambda_1.flatten()
    Lambda_1_sort=np.sort(np.abs(Lambda_1_flat))
    Lambda_4_flat=Lambda_4.flatten()
    Lambda_4_sort=np.sort(np.abs(Lambda_4_flat))
    
    num=np.arange(np.size(Sigma_1_sort))
    num_lbd=np.arange(p['n_area'])
    fig, ax = plt.subplots(2,3)
    f=ax[0,0].plot(num,Sigma_1_sort,'-o')
    ax[0,0].set_title('Sigma_1_sort')
    ax[0,0].set_yscale('log')
    ax[0,0].set_ylim([0,1])
    ax[0,0].set_ylabel('element value')
    f=ax[0,1].plot(num,Sigma_2_sort,'-o')
    ax[0,1].set_title('Sigma_2_sort')
    ax[0,1].set_yscale('log')
    ax[0,1].set_ylim([0,1])
    f=ax[0,2].plot(num_lbd,Lambda_1_sort[Lambda_1_sort>0],'-o')
    ax[0,2].set_title('Lambda_1_sort')
    ax[0,2].set_yscale('log')
    ax[0,2].set_ylim([0,1])
    f=ax[1,0].plot(num,Sigma_3_sort,'-o')
    ax[1,0].set_title('Sigma_3_sort')
    ax[1,0].set_yscale('log')
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_ylabel('element value')
    f=ax[1,1].plot(num,Sigma_4_sort,'-o')
    ax[1,1].set_title('Sigma_4_sort')
    ax[1,1].set_yscale('log')
    ax[1,1].set_ylim([0,1])
    f=ax[1,2].plot(num_lbd,Lambda_4_sort[Lambda_4_sort>0],'-o')
    ax[1,2].set_title('Lambda_4_sort')
    ax[1,2].set_yscale('log')
    ax[1,2].set_ylim([1e-1,1])
    
    fig, ax = plt.subplots(1,3)
    diff_lambda1=np.zeros_like(Lambda_1)
    for i in np.arange(p['n_area']):
        for j in np.arange(p['n_area']):
           diff_lambda1[i,j]=Lambda_1[i,i]-Lambda_1[j,j]
           
    f=ax[0].plot(np.sort(diff_lambda1.flatten()),'-ro',markersize=1)
    ax[0].set_title(r'$diff\ \ \Lambda_1$')
    ax[0].set_yscale('log')
    ax[0].set_ylim([1e-5,0.1])
    ax[0].set_ylabel('element value')
    ax[0].set_xlabel('element index')
    
    f=ax[1].plot(num,Sigma_1_sort,'-go',markersize=1)
    ax[1].set_title(r'$\Sigma_1$')
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e-5,0.1])
    ax[1].set_xlabel('element index')
    
    f=ax[2].plot(num,Sigma_2_sort,'-bo',markersize=1)
    ax[2].set_title(r'$\Sigma_2$')
    ax[2].set_yscale('log')
    ax[2].set_ylim([1e-5,0.1])
    ax[2].set_xlabel('element index')
      
    fig, ax = plt.subplots()
    ax.plot(np.sort(np.abs(np.real(diff_lambda1.flatten()))),'o',markersize=3,label=r'$\Lambda_1$')
    ax.plot(num,Sigma_1_sort,'o',markersize=3,label=r'$\Sigma_1$')
    ax.plot(num,Sigma_2_sort,'o',markersize=3,label=r'$\Sigma_2$')
    ax.set_yscale('log')
    ax.set_ylim([0,1])
    ax.set_ylabel('values')
    ax.set_xlabel('index')
    ax.legend()

    
    
    #--------------------------------------------------------------------------
    #compare the gap of eigenvalues and the off-diagnal elements
    #--------------------------------------------------------------------------
    temp_lbd=np.diag(Lambda_1)
    dif_lbd=np.zeros(p['n_area']**2)
    count=0
    for i in np.arange(p['n_area']):
        for j in np.arange(i+1,p['n_area']):
            dif_lbd[count]=np.abs(temp_lbd[i]-temp_lbd[j])
            count=count+1
            
    fig, ax = plt.subplots()
    #f=ax.hist(np.diff(np.diag(Lambda_1)),200,facecolor='r')
    f=ax.hist(dif_lbd[dif_lbd>0],100,facecolor='r',alpha=0.5,label='eigval diff')
    Sigma_nonzero=Sigma_1.flatten()
    Sigma_nonzero=Sigma_nonzero[Sigma_nonzero>0]
    f=ax.hist(Sigma_nonzero,200,facecolor='b',alpha=0.5,label='sigma_1')
    ax.set_yscale('log')
    ax.set_xlabel('value')
    ax.set_ylabel('counts')
    ax.legend()
    print('mean_sigma1=',np.mean(Sigma_nonzero))
#------------------------------------------------------------------------------
#approximation of the eigenvectors with components u and v
#------------------------------------------------------------------------------
    
#------------------------------first u case------------------------------------
    #eigvals_norder, eigvecs_norder = np.linalg.eig(Lambda_1+Sigma_1)    #TEST TEST TEST np.linalg.eig(Lambda_1+Sigma_1)
    
    # ind=np.argsort(-np.real(eigvals_norder))
    # id_mat=np.eye(p['n_area'])
    # per_mat=id_mat.copy()
    # for i in np.arange(p['n_area']):
    #     per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    # eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    # eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    # eigvals=np.diag(eigmat) 
    
    # u_1=eigvecs.copy()
    # v_1=-np.linalg.inv(Lambda_4+Sigma_4-np.diag(eigvals))@Sigma_3@eigvecs
    # lbd_1=eigvals.copy()
    
    # r_E_1=(np.eye(p['n_area'])+A@B)@u_1-A@v_1
    # r_I_1=-B@u_1+v_1

    #2021-3-20 compute the perturbated eigenvalues and eigenvectors of Lambda_1 + Sigma_1 TEST TEST TEST 2021-3-20
    eigvals_norder, sec_order_eigvals,eigvecs_norder,sec_order_eigvecs=eig_approx(Lambda_1,Sigma_1)
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 
    
    u_1=eigvecs.copy()
    v_1=-np.linalg.inv(Lambda_4+Sigma_4-np.diag(eigvals))@Sigma_3@eigvecs
    lbd_1=eigvals.copy()
    
    r_E_1=(np.eye(p['n_area'])+A@B)@u_1-A@v_1
    r_I_1=-B@u_1+v_1
    

#------------------------------second u case-----------------------------------    
    eigvals_norder, eigvecs_norder = np.linalg.eig(Lambda_4+Sigma_4)
    
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 

    u_2=np.zeros((p['n_area'],p['n_area']))
    v_2=eigvecs.copy()
    lbd_2=eigvals.copy()
    
    r_E_2=(np.eye(p['n_area'])+A@B)@u_2-A@v_2
    r_I_2=-B@u_2+v_2
    
#-----------------------------plot u and v-------------------------------------  
    eig_val=(0+0j)*np.zeros(2*p['n_area'])
    eig_val[0:p['n_area']]=lbd_1
    eig_val[p['n_area']:]=lbd_2
     
    eig_vec=(0+0j)*np.zeros((2*p['n_area'],2*p['n_area']))
    eig_vec[0:p['n_area'],0:p['n_area']]=u_1
    eig_vec[p['n_area']:,0:p['n_area']]=v_1
    eig_vec[0:p['n_area'],p['n_area']:]=u_2
    eig_vec[p['n_area']:,p['n_area']:]=v_2
    
    for i in np.arange(2*p['n_area']):
        eig_vec[:,i]=eig_vec[:,i]/np.linalg.norm(eig_vec[:,i])
        
    temp_eig=np.real(eig_val)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eig_vec),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization-u-v')

    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(eigvals_s[::1])
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
#--------------------------------plot R_E R_I----------------------------------   
    eig_val=(0+0j)*np.zeros(2*p['n_area'])
    eig_val[0:p['n_area']]=lbd_1
    eig_val[p['n_area']:]=lbd_2
     
    eig_vec=(0+0j)*np.zeros((2*p['n_area'],2*p['n_area']))
    eig_vec[0:p['n_area'],0:p['n_area']]=r_E_1
    eig_vec[p['n_area']:,0:p['n_area']]=r_I_1
    eig_vec[0:p['n_area'],p['n_area']:]=r_E_2
    eig_vec[p['n_area']:,p['n_area']:]=r_I_2
    
    for i in np.arange(2*p['n_area']):
        eig_vec[:,i]=eig_vec[:,i]/np.linalg.norm(eig_vec[:,i])
        
    temp_eig=np.real(eig_val)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eig_vec),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization_rE_rI')

    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(eigvals_s[::1])
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    
    #--------------------------------------------------------------------------
    #eigenmode decomposition
    #--------------------------------------------------------------------------
    eigvals_norder, eigvecs_norder = np.linalg.eig(W1_EI)
    ind=np.argsort(-np.real(eigvals_norder))
    id_mat=np.eye(2*p['n_area'])
    per_mat=id_mat.copy()
    for i in np.arange(2*p['n_area']):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigmat=per_mat@(np.diag(eigvals_norder))@(np.linalg.inv(per_mat))
    eigvecs=eigvecs_norder@(np.linalg.inv(per_mat))
    eigvals=np.diag(eigmat) 
        
    temp_eig=np.real(eigvals)
    eigvals_s=np.zeros_like(temp_eig)
    for i in range(len(temp_eig)):
        eigvals_s[i]=format(temp_eig[i],'.5f')
        
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigvecs),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W matrix eigenvector visualization')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(eigvals_s[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')       
    #run_stimulus(p,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=MACAQUE_CASE)


    # #--------------------------------------------------------------------------
    # #eigenmode resort 2021-3-20
    # #--------------------------------------------------------------------------
    # eigvecs_ori=np.abs(eigvecs)  #from original
    # eigvecs_app=np.abs(eig_vec)    #from approximation
    # index_list=[]
    # max_inprod_list=[]
    
    
    # length=len(temp_eig)
    # for i in range(length):
    #     max_inprod=-1
    #     for j in range(length):
    #         # print('j=',j)
    #         # print(index_list)
    #         # j not in index_list
    #         if j not in index_list: 
    #            temp_max_inprod = np.dot(eigvecs_app[:,length-i-1], eigvecs_ori[:,length-j-1])
    #            if temp_max_inprod >max_inprod:
    #                max_inprod=temp_max_inprod
    #                max_index=j
    #     max_inprod_list.append(max_inprod)
    #     index_list.append(max_index)
    # print(index_list)    
    # print(max_inprod_list)
    
    # plt.figure(figsize=(5,7))        
    # ax = plt.axes()
    # plt.bar(np.arange(int(length/2)),max_inprod_list[int(length/2):],width = 1,color='b')
    # # plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    # plt.ylabel('similarity')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # eigvecs_reorder=np.zeros_like(eigvecs_ori)
    # for i in np.arange(2*p['n_area']):        
    #     eigvecs_reorder[:,length-i-1]=eigvecs_ori[:,length-1-index_list[i]]
        
    # fig, ax = plt.subplots(figsize=(20,10))
    # f=ax.pcolormesh(np.abs(eigvecs_reorder),cmap='hot')
    # fig.colorbar(f,ax=ax,pad=0.1)
    # ax.set_title('W matrix eigenvector visualization reorder')
    
    # x = np.arange(2*p['n_area']) # xticks
    # y = np.arange(2*p['n_area']) # yticks
    # xlim = (0,2*p['n_area'])
    # ylim = (0,2*p['n_area'])
    
    # yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    # yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # # set original ticks and ticklabels
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.set_xticks(x[::1])
    # ax.invert_xaxis()
        
    # ax.set_xticklabels(eigvals_s[::1])
    # ax.set_yticks(y[::2])
    # ax.set_yticklabels(yticklabels_even)
    # ax.invert_yaxis()
    # # rotate xticklabels to 90 degree
    # plt.setp(ax.get_xticklabels(), rotation=90)
    
    # # second y axis
    # ax3 = ax.twinx()
    # ax3.set_ylim(ylim)
    # ax3.set_yticks(y[1::2])
    # ax3.set_yticklabels(yticklabels_odd)
    # ax3.invert_yaxis()   
    # #fig.savefig('result/full_eigenmode.pdf')       
    # #run_stimulus(p,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=MACAQUE_CASE)


    #--------------------------------------------------------------------------
    #egienmatrix approximation resort 2021-3-20
    #--------------------------------------------------------------------------
    eigvecs_ori=np.abs(eigvecs)  #from original
    eigvecs_app=np.abs(eig_vec)    #from approximation
    index_list=[]
    max_inprod_list=[]
    
    length=len(temp_eig)
    for i in range(length):
        max_inprod=-1
        for j in range(length):
            # print('j=',j)
            # print(index_list)
            # j not in index_list
            if j not in index_list: 
               temp_max_inprod = np.dot(eigvecs_ori[:,length-i-1], eigvecs_app[:,length-j-1])
               if temp_max_inprod >max_inprod:
                   max_inprod=temp_max_inprod
                   max_index=j
        max_inprod_list.append(max_inprod)
        index_list.append(max_index)
    print(index_list)    
    print(max_inprod_list)
    
    plt.figure(figsize=(20,5))        
    ax = plt.axes()
    plt.bar(np.arange(length),max_inprod_list,color='k',alpha=0.5)
    # plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('similarity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    eigvecs_reorder=np.zeros_like(eigvecs_app)
    for i in np.arange(2*p['n_area']):        
        eigvecs_reorder[:,length-i-1]=eigvecs_app[:,length-1-index_list[i]]
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(eigvecs_reorder),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('W approximation eigenvector visualization reorder')
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(eigvals_s[::1])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
      
            
#plot the localization of an eigenvector plot ipr_theta_tau map
def quant_local(eigmat_t,tau_t,d_mat_t):
    eigmat=eigmat_t.copy()
    tau=tau_t.copy()
    d_mat=d_mat_t.copy()
    
    lens=len(tau)        
    theta=np.zeros(lens)
    ipr=np.zeros(lens)
    for i in range(lens):
        theta[i]=THETA(eigmat[:,i],d_mat)
        ipr[i]=IPR(eigmat[:,i])  
    
    fig,ax=plt.subplots()
    f=ax.scatter(np.flip(ipr),np.flip(theta),30,np.log(np.flip(tau)),cmap='jet',alpha=0.8)
    ax.set_ylim([0,1.1])
    ax.set_xlim(xmin=0)
    #ax.set_xlim([-3,0])
    plt.xlabel('IPR')
    plt.ylabel(r'$\theta$')
    cbar=plt.colorbar(f)
    cbar.set_label(r'$log(\tau)$', rotation=90)
    fig.savefig('result/quantify_localization.pdf')    

    fig,ax=plt.subplots(2,3)
    ax[0,0].hist(theta,20,facecolor='r',alpha=0.5)
    ax[0,0].set_xlabel('theta')
    ax[0,0].set_xlim([0,1])
    ax[0,1].hist(ipr,20,facecolor='b',alpha=0.5)
    ax[0,1].set_xlabel('ipr')
    ax[0,1].set_xlim([0,1])
    ax[0,2].hist(tau,20,facecolor='g',alpha=0.5)
    ax[0,2].set_xlabel('tau')
    ax[0,2].set_xlim(xmin=0)    
                        
    ax[1,0].hist(theta[:int(lens/2)],20,facecolor='r',alpha=0.5)
    ax[1,0].set_xlabel('theta')
    ax[1,0].set_xlim([0,1])
    ax[1,1].hist(ipr[:int(lens/2)],20,facecolor='b',alpha=0.5)
    ax[1,1].set_xlabel('ipr')
    ax[1,1].set_xlim([0,1])
    ax[1,2].hist(tau[:int(lens/2)],20,facecolor='g',alpha=0.5)
    ax[1,2].set_xlabel('tau')
    ax[1,2].set_xlim(xmin=0)
                           
#generate distance matrix of macaque and marmoset networks    
def generate_dist_matrix(p_t,MACAQUE_CASE=0,CONSENSUS_CASE=0):
    
    print('In generate_dist_matrix, CONSENSUS_CASE=',CONSENSUS_CASE)
    p=p_t.copy()
    p['n_tau']=2*p['n_area']
    
    dist_mat=np.zeros((p['n_area']+1,p['n_area']+1))
    full_dist_mat=np.zeros((2*p['n_area'],2*p['n_area']))  #computed as EE, EI, IE, II blocks
        
    if MACAQUE_CASE:
        if CONSENSUS_CASE==0:
            datafile='macaque_distance_data.pkl'
        else:
            datafile='macaque_distance_data_consensus.pkl'    
    else:
        if CONSENSUS_CASE==0:
            datafile='marmoset_distance_data.pkl'
        else:
            datafile='marmoset_distance_data_consensus.pkl'
            
    with open(datafile,'rb') as f:
        p2 = pickle.load(f, encoding='latin1')
    
    for i in np.arange(p['n_area']):
        for j in np.arange(p['n_area']):
            target_id=p2['areas'].index(p['areas'][i])
            source_id=p2['areas'].index(p['areas'][j])
            dist_mat[i,j]=p2['distance_matrix'][target_id,source_id]
            full_dist_mat[i,j]=p2['distance_matrix'][target_id,source_id]
            full_dist_mat[i+p['n_area'],j]=p2['distance_matrix'][target_id,source_id]
            full_dist_mat[i,j+p['n_area']]=p2['distance_matrix'][target_id,source_id]
            full_dist_mat[i+p['n_area'],j+p['n_area']]=p2['distance_matrix'][target_id,source_id]

    dist_mat[-1,0:p['n_area']]=np.ones(p['n_area'])*np.mean(p2['distance_matrix'])
    dist_mat[0:p['n_area'],-1]=np.ones(p['n_area'])*np.mean(p2['distance_matrix'])  
    return dist_mat,full_dist_mat

#plot theta-IPR diagram for macaque or marmoset network   
def time_scale_localization_theta_IPR_diagram(p_t,eigVecs_reorder_t,tau_reorder_t,dist_mat_t,full_dist_mat_t,FULL_CASE=1,EXC_CASE=0,MACAQUE_CASE=1):    
    
    #EXC_CASE=1 focuses on the time scale localization at the E populations, all the I populations are viewed as a single node, mean distance is used to represent the effective distance to all the E nodes, comp_eigvec is calculated to incorporate all I populations' eigen-elements into a single value
    #EXC_CASE=0 focuses on the time scale localization at the I populations, others are similar to the case of EXC_CASE=1
    #FULL_CASE=1 focuses on the time scale localization at both the E and I populations
    
    p=p_t.copy()
    eigVecs_reorder=eigVecs_reorder_t.copy()
    tau_reorder=tau_reorder_t.copy()
    dist_mat=dist_mat_t.copy()
    full_dist_mat=full_dist_mat_t.copy()
    
    if FULL_CASE+EXC_CASE>1:
        raise SystemExit('Conflict of case parameter setting!')
    
    comp_eigvec=np.zeros((p['n_area']+1,2*p['n_area']))
       
    if EXC_CASE:
        comp_eigvec[0:p['n_area'],:]=eigVecs_reorder[0:p['n_area'],]
    else: 
        comp_eigvec[0:p['n_area'],:]=eigVecs_reorder[p['n_area']:,:]
    
    rest_v=np.zeros(2*p['n_area'])
    for i in np.arange(2*p['n_area']):
        rest_v[i]=np.sqrt(1-comp_eigvec[:,i].dot(comp_eigvec[:,i]))
        comp_eigvec[-1,i]=rest_v[i]  
        #print(np.linalg.norm(comp_eigvec[:,i],ord=2))
    
    if FULL_CASE:
        quant_local(eigVecs_reorder,tau_reorder,full_dist_mat)
        ind=np.where(tau_reorder==max(tau_reorder))
        network_graph_plot(p,eigVecs_reorder[:p['n_area'],ind],MACAQUE_CASE=MACAQUE_CASE)
        
    else:    
        quant_local(comp_eigvec,tau_reorder,dist_mat)
    
        fig, ax = plt.subplots(figsize=(20,10))
        f=ax.pcolormesh(comp_eigvec,cmap='hot')
        fig.colorbar(f,ax=ax,pad=0.1)
        
        x = np.arange(2*p['n_area']) # xticks
        y = np.arange(p['n_area']+1) # yticks
        xlim = (0,2*p['n_area'])
        ylim = (0,p['n_area']+1)
        
        yticklabels_odd=p['areas'][1::2]
        yticklabels_even=p['areas'][::2]
        yticklabels_odd.append('RES')
        # set original ticks and ticklabels
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(x[::2])
        ax.invert_xaxis()
           
        ax.set_xticklabels(tau_reorder[::2])
        ax.set_yticks(y[::2])
        ax.set_yticklabels(yticklabels_even)
        ax.invert_yaxis()
        # rotate xticklabels to 90 degree
        plt.setp(ax.get_xticklabels(), rotation=90)
        
        # second y axis
        ax3 = ax.twinx()
        ax3.set_ylim(ylim)
        ax3.set_yticks(y[1::2])
        ax3.set_yticklabels(yticklabels_odd)
        ax3.invert_yaxis()   
        ax.set_title('eigenvector visualization')
        fig.savefig('result/slowest_eigenmode2.pdf')    
        
        
#run simulation of netowrk responses    
def run_stimulus(p_t,VISUAL_INPUT=1,TOTAL_INPUT=0,T=6000,PULSE_INPUT=1,MACAQUE_CASE=1,GATING_PATHWAY=0,CONSENSUS_CASE=0, stim_area='V1',plot_Flag=1):
        
        if VISUAL_INPUT:
            area_act = stim_area   #V1
        else:
            if MACAQUE_CASE:
                area_act='2'
            else:
                area_act = 'AuA1'
        print('Running network with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

        #---------------------------------------------------------------------------------
        # Redefine Parameters
        #---------------------------------------------------------------------------------

        p=p_t.copy()

        # Definition of combined parameters

        local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
        local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)
        
        fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
        fln_scaled_inh = (p['inh_scale'] * p['fln_mat'].T).T
        #---------------------------------------------------------------------------------
        # Simulation Parameters
        #---------------------------------------------------------------------------------

        dt = 0.05  # ms
        if PULSE_INPUT:
            T = T
        else:                
            T = T
        
        t_plot = np.linspace(0, T, int(T/dt)+1)
        n_t = len(t_plot)  
        
        r_exc_base = 10
        # From target background firing inverts background inputs
        r_exc_tgt = r_exc_base * np.ones(p['n_area'])  # 10   
        r_inh_tgt = 35 * np.ones(p['n_area'])

        longrange_E = np.dot(fln_scaled,r_exc_tgt)
        longrange_I = np.dot(fln_scaled_inh,r_exc_tgt)

        I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                                 + p['beta_exc']*p['muEE']*longrange_E)
        I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                                 + p['beta_inh']*p['muIE']*longrange_I)

        # Set stimulus input
        I_stim_exc = np.zeros((n_t,p['n_area']))

        area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
        
        # #==============TEST TEST TEST XXXXX DEBUG DEBUG=========================
        # area_stim_idx2 = p['areas'].index('9/46d') # Index of stimulated area
        
        # if PULSE_INPUT:
        #     time_idx = (t_plot>100) & (t_plot<=350) # original
        #     # time_idx = (t_plot>100) & (t_plot<=T-1)
        #     I_stim_exc[time_idx, area_stim_idx] = 41.187
        #     # I_stim_exc[time_idx, area_stim_idx2] = 41.187
        # else:
        #     for i in range(p['n_area']):
        #         I_stim_exc[:,i]=gaussian_noise(0,1e-5,n_t)        
        #     I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t) #2,0.5
        
        if PULSE_INPUT:
            time_idx = (t_plot>200) & (t_plot<=400)
            I_stim_exc[time_idx, area_stim_idx] = 41.187
        else:
            if TOTAL_INPUT:
                for i in range(p['n_area']):
                    I_stim_exc[:, i] = gaussian_noise(0,1e-5,n_t)
            else:
                for i in range(p['n_area']):
                    I_stim_exc[:,i] = gaussian_noise(0,1e-5,n_t)        
                I_stim_exc[:, area_stim_idx] = gaussian_noise(0,0.5,n_t)
        
        #---------------------------------------------------------------------------------
        # Storage
        #---------------------------------------------------------------------------------

        r_exc = np.zeros((n_t,p['n_area']))
        r_inh = np.zeros((n_t,p['n_area']))

        #---------------------------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------------------------
        fI = lambda x : x*(x>0)
        # fI = lambda x : x
        #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)  #for GATING_PATHWAY==1 only
        
        # Set activity to background firing
        r_exc[0] = r_exc_tgt
        r_inh[0] = r_inh_tgt
        
        #---------------------------------------------------------------------------------
        # Running the network
        #---------------------------------------------------------------------------------

        for i_t in range(1, n_t):
            longrange_E = np.dot(fln_scaled,r_exc[i_t-1])
            longrange_I = np.dot(fln_scaled_inh,r_exc[i_t-1])
            I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                     p['beta_exc'] * p['muEE'] * longrange_E +
                     I_bkg_exc + I_stim_exc[i_t])

            I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                     p['beta_inh'] * p['muIE'] * longrange_I + I_bkg_inh)
            
            if GATING_PATHWAY:
                d_local_EI=np.zeros_like(local_EI)
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V4','8m']
                 
                for name in area_name_list:
                    area_idx=p['areas'].index(name)
                    d_local_EI[area_idx]=-local_EI[area_idx]*0.07  #0.1
                
                if I_stim_exc[i_t,area_stim_idx]>10:
                    I_exc=I_exc+d_local_EI*r_inh[i_t-1]
                
            d_r_exc = -r_exc[i_t-1] + fI(I_exc)
            d_r_inh = -r_inh[i_t-1] + fI(I_inh)

            r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
            r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']
        
        #---------------------------------------------------------------------------------
        # Plotting step input results
        #---------------------------------------------------------------------------------
        if CONSENSUS_CASE==0:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
                else:
                    area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
                else:
                    area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
        else:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m 8l 8r','5','TEO TEOm','F4','9/46d 46d','TEpd TEa/ma TEa/mp','F7']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','PEC PE','LIP','PGM','A32 A32V','A6DR','A6Va A6Vb']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
        max_rate=np.max(r_exc-r_exc_base,axis=0)
        decay_time=np.zeros(p['n_area'])
        if plot_Flag:
            area_idx_list=[-1]
            for name in area_name_list:
                area_idx_list=area_idx_list+[p['areas'].index(name)]
            #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
            
            f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(12, 12))
            
            clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
            c_color=0
            for ax, area_idx in zip(ax_list, area_idx_list):
                if area_idx < 0:
                    y_plot = I_stim_exc[:, area_stim_idx].copy()
                    z_plot = np.zeros_like(y_plot)
                    txt = 'Input'
    
                else:
                    y_plot = r_exc[:,area_idx].copy()
                    z_plot = r_inh[:,area_idx].copy()
                    txt = p['areas'][area_idx]
    
                if PULSE_INPUT:
                    y_plot = y_plot - y_plot.min()
                    # y_plot = y_plot - 10
                    z_plot = z_plot - z_plot.min()
                    ax.plot(t_plot, y_plot,color='k', linewidth=3)
                    #ax.plot(t_plot, z_plot,'--',color='b')
                else:
                    #ax.plot(t_plot, y_plot,color='r')
                    ax.plot(t_plot[0:10000], y_plot[-1-10000:-1],color='r', linewidth=2)
                    # ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')
                    
                # ax.plot(t_plot, y_plot,color=clist[0][c_color])
                # ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
                c_color=c_color+1
                ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=18)
    
                if PULSE_INPUT:
                    ax.set_yticks([0,y_plot.max()])
                    # ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
                    ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max())], fontsize=18)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                #ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                plt.xticks(fontsize=18)

            f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=18)
            ax.set_xlabel('Time (ms)', fontsize=18)    
            
            if PULSE_INPUT:
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                plt.figure(figsize=(12, 8))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    # if area_idx < 0:
                    #     continue
                    # else:
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    plt.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    c_color=c_color+1
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('Time (ms)', fontsize=18)

                decay_time=np.zeros(p['n_area'])
                # fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,20))
                fig, ax=plt.subplots(figsize=(15,10))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()-t_plot[get_posi]
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    # ax1.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    p_end=np.where(y_plot_cut>1/np.e)[0][-1]
                    decay_time[c_color]=p_end*dt
                    c_color=c_color+1
                
                # ax1.set_xlabel('time (ms)')
                # ax1.set_ylabel('normalized response')
                
        
                ax.bar(np.arange(len(p['areas'])),decay_time,width = 1,color=clist[0])
                ax.set_xticks(np.arange(len(p['areas'])))
                ax.set_xticklabels(p['areas'],rotation=90, fontsize=16)
                ax.set_yscale('log')
                plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0, fontsize=20)
                ax.set_ylabel('Decay time (ms)',fontsize=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                decay_time=np.zeros(p['n_area']) 
            
            max_rate=np.max(r_exc-r_exc_base,axis=0)
            # network_graph_plot(p,max_rate,MACAQUE_CASE=MACAQUE_CASE)
            
            fig,ax=plt.subplots(figsize=(15,10))
            ax.plot(np.arange(len(p['areas'])), max_rate,'-o')
            ax.set_xticks(np.arange(len(p['areas'])))
            ax.set_xticklabels(p['areas'],rotation=90,fontsize=16)
            ax.set_yscale('log')
            ax.set_ylabel('Max Firing Rate',fontsize=16)
            # ax.set_xlabel('hierarchy values')

        if plot_Flag:
            return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate, f
        else:
            return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate
        # return I_stim_exc, r_exc, r_inh, area_stim_idx
        # return max_rate



#run simulation of netowrk responses with Oscillatory Input
def run_stimulus_osc(p_t,VISUAL_INPUT=1,TOTAL_INPUT=0,T=6000,PULSE_INPUT=0,OSC_INPUT=1,f_OSC=np.sin, MACAQUE_CASE=1,GATING_PATHWAY=0,CONSENSUS_CASE=0, stim_area='V1',plot_Flag=1):
        
        if VISUAL_INPUT:
            area_act = stim_area   #V1
        else:
            if MACAQUE_CASE:
                area_act='2'
            else:
                area_act = 'AuA1'
        print('Running network with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

        #---------------------------------------------------------------------------------
        # Redefine Parameters
        #---------------------------------------------------------------------------------

        p=p_t.copy()

        # Definition of combined parameters

        local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
        local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)
        
        fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
        fln_scaled_inh = (p['inh_scale'] * p['fln_mat'].T).T
        #---------------------------------------------------------------------------------
        # Simulation Parameters
        #---------------------------------------------------------------------------------

        dt = 0.05  # ms
        if PULSE_INPUT:
            T = T
        else:                
            T = T
        
        t_plot = np.linspace(0, T, int(T/dt)+1)
        n_t = len(t_plot)  
        
        r_exc_base = 10
        # From target background firing inverts background inputs
        r_exc_tgt = r_exc_base * np.ones(p['n_area'])  # 10   
        r_inh_tgt = 35 * np.ones(p['n_area'])

        longrange_E = np.dot(fln_scaled,r_exc_tgt)
        longrange_I = np.dot(fln_scaled_inh,r_exc_tgt)

        I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                                 + p['beta_exc']*p['muEE']*longrange_E)
        I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                                 + p['beta_inh']*p['muIE']*longrange_I)

        # Set stimulus input
        I_stim_exc = np.zeros((n_t,p['n_area']))

        area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
        
        # #==============TEST TEST TEST XXXXX DEBUG DEBUG=========================
        # area_stim_idx2 = p['areas'].index('9/46d') # Index of stimulated area
        
        # if PULSE_INPUT:
        #     time_idx = (t_plot>100) & (t_plot<=350) # original
        #     # time_idx = (t_plot>100) & (t_plot<=T-1)
        #     I_stim_exc[time_idx, area_stim_idx] = 41.187
        #     # I_stim_exc[time_idx, area_stim_idx2] = 41.187
        # else:
        #     for i in range(p['n_area']):
        #         I_stim_exc[:,i]=gaussian_noise(0,1e-5,n_t)        
        #     I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t) #2,0.5
        
        if PULSE_INPUT:
            time_idx = (t_plot>200) & (t_plot<=400)
            I_stim_exc[time_idx, area_stim_idx] = 41.187

        if OSC_INPUT: # t_plot unit: ms
            I_stim_exc[:, area_stim_idx] = f_OSC(t_plot) # np.sin(2*np.pi*0.5*t_plot/1000) +  np.sin(2*np.pi*(1.5*t_plot/1000+ 0.25)) + 2

        if TOTAL_INPUT:
            for i in range(p['n_area']):
                I_stim_exc[:, i] += gaussian_noise(0,1e-5,n_t)
        
        #---------------------------------------------------------------------------------
        # Storage
        #---------------------------------------------------------------------------------

        r_exc = np.zeros((n_t,p['n_area']))
        r_inh = np.zeros((n_t,p['n_area']))

        #---------------------------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------------------------
        fI = lambda x : x*(x>0)
        # fI = lambda x : x
        #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)  #for GATING_PATHWAY==1 only
        
        # Set activity to background firing
        r_exc[0] = r_exc_tgt
        r_inh[0] = r_inh_tgt
        
        #---------------------------------------------------------------------------------
        # Running the network
        #---------------------------------------------------------------------------------

        for i_t in range(1, n_t):
            longrange_E = np.dot(fln_scaled,r_exc[i_t-1])
            longrange_I = np.dot(fln_scaled_inh,r_exc[i_t-1])
            I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                     p['beta_exc'] * p['muEE'] * longrange_E +
                     I_bkg_exc + I_stim_exc[i_t])

            I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                     p['beta_inh'] * p['muIE'] * longrange_I + I_bkg_inh)
            
            if GATING_PATHWAY:
                d_local_EI=np.zeros_like(local_EI)
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V4','8m']
                 
                for name in area_name_list:
                    area_idx=p['areas'].index(name)
                    d_local_EI[area_idx]=-local_EI[area_idx]*0.07  #0.1
                
                if I_stim_exc[i_t,area_stim_idx]>10:
                    I_exc=I_exc+d_local_EI*r_inh[i_t-1]
                
            d_r_exc = -r_exc[i_t-1] + fI(I_exc)
            d_r_inh = -r_inh[i_t-1] + fI(I_inh)

            r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
            r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']
        
        #---------------------------------------------------------------------------------
        # Plotting step input results
        #---------------------------------------------------------------------------------
        if CONSENSUS_CASE==0:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
                else:
                    area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
                else:
                    area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
        else:
            if MACAQUE_CASE:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V4','8m 8l 8r','5','TEO TEOm','F4','9/46d 46d','TEpd TEa/ma TEa/mp','F7']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
            else:
                if VISUAL_INPUT:
                    area_name_list = ['V1','V2','V4','PEC PE','LIP','PGM','A32 A32V','A6DR','A6Va A6Vb']
                else:
                    raise SystemExit('Must give Visual input to networks under consensus map!')
                    
        max_rate=np.max(r_exc-r_exc_base,axis=0)
        decay_time=np.zeros(p['n_area'])
        if plot_Flag:
            area_idx_list=[-1]
            for name in area_name_list:
                area_idx_list=area_idx_list+[p['areas'].index(name)]
            #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
            
            f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(12, 12))
            
            clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
            c_color=0
            for ax, area_idx in zip(ax_list, area_idx_list):
                if area_idx < 0:
                    y_plot = I_stim_exc[:, area_stim_idx].copy()
                    z_plot = np.zeros_like(y_plot)
                    txt = 'Input'
    
                else:
                    y_plot = r_exc[:,area_idx].copy()
                    z_plot = r_inh[:,area_idx].copy()
                    txt = p['areas'][area_idx]
    
                if PULSE_INPUT:
                    y_plot = y_plot - y_plot.min()
                    # y_plot = y_plot - 10
                    z_plot = z_plot - z_plot.min()
                    ax.plot(t_plot, y_plot,color='k', linewidth=3)
                    #ax.plot(t_plot, z_plot,'--',color='b')
                else:
                    #ax.plot(t_plot, y_plot,color='r')
                    ax.plot(t_plot, y_plot,color='r', linewidth=2)
                    # ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')
                    
                # ax.plot(t_plot, y_plot,color=clist[0][c_color])
                # ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
                c_color=c_color+1
                ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=18)
    
                if PULSE_INPUT:
                    ax.set_yticks([0,y_plot.max()])
                    # ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
                    ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max())], fontsize=18)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                #ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                plt.xticks(fontsize=18)

            f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=18)
            ax.set_xlabel('Time (ms)', fontsize=18)    
            
            if PULSE_INPUT:
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                plt.figure(figsize=(12, 8))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    # if area_idx < 0:
                    #     continue
                    # else:
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    plt.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    c_color=c_color+1
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('Time (ms)', fontsize=18)

                decay_time=np.zeros(p['n_area'])
                # fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,20))
                fig, ax=plt.subplots(figsize=(15,10))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()-t_plot[get_posi]
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    # ax1.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    p_end=np.where(y_plot_cut>1/np.e)[0][-1]
                    decay_time[c_color]=p_end*dt
                    c_color=c_color+1
                
                # ax1.set_xlabel('time (ms)')
                # ax1.set_ylabel('normalized response')
                
        
                ax.bar(np.arange(len(p['areas'])),decay_time,width = 1,color=clist[0])
                ax.set_xticks(np.arange(len(p['areas'])))
                ax.set_xticklabels(p['areas'],rotation=90, fontsize=16)
                ax.set_yscale('log')
                plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0, fontsize=20)
                ax.set_ylabel('Decay time (ms)',fontsize=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                decay_time=np.zeros(p['n_area']) 
            
            max_rate=np.max(r_exc-r_exc_base,axis=0)
            # network_graph_plot(p,max_rate,MACAQUE_CASE=MACAQUE_CASE)
            
            fig,ax=plt.subplots(figsize=(15,10))
            ax.plot(np.arange(len(p['areas'])), max_rate,'-o')
            ax.set_xticks(np.arange(len(p['areas'])))
            ax.set_xticklabels(p['areas'],rotation=90,fontsize=16)
            ax.set_yscale('log')
            ax.set_ylabel('Max Firing Rate',fontsize=16)
            # ax.set_xlabel('hierarchy values')

        if plot_Flag:
            return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate, f
        else:
            return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate
        # return I_stim_exc, r_exc, r_inh, area_stim_idx
        # return max_rate


#run simulation of netowrk responses with current input also reported    
def run_stimulus_wi_input(p_t,VISUAL_INPUT=1,TOTAL_INPUT=0,T=6000,PULSE_INPUT=1,MACAQUE_CASE=1,GATING_PATHWAY=0,CONSENSUS_CASE=0, stim_area='V1',plot_Flag=0):
        
        if VISUAL_INPUT:
            area_act = stim_area   #V1
        else:
            if MACAQUE_CASE:
                area_act='2'
            else:
                area_act = 'AuA1'
        print('Running network with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

        #---------------------------------------------------------------------------------
        # Redefine Parameters
        #---------------------------------------------------------------------------------

        p=p_t.copy()

        # Definition of combined parameters

        local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
        local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)
        
        fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
        fln_scaled_inh = (p['inh_scale'] * p['fln_mat'].T).T
        #---------------------------------------------------------------------------------
        # Simulation Parameters
        #---------------------------------------------------------------------------------

        dt = 0.05  # ms
        if PULSE_INPUT:
            T = T
        else:                
            T = T
        
        t_plot = np.linspace(0, T, int(T/dt)+1)
        n_t = len(t_plot)  
        
        r_exc_base = 10
        # From target background firing inverts background inputs
        r_exc_tgt = r_exc_base * np.ones(p['n_area'])  # 10   
        r_inh_tgt = 35 * np.ones(p['n_area'])

        longrange_E = np.dot(fln_scaled,r_exc_tgt)
        longrange_I = np.dot(fln_scaled_inh,r_exc_tgt)

        I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                                 + p['beta_exc']*p['muEE']*longrange_E)
        I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                                 + p['beta_inh']*p['muIE']*longrange_I)

        # Set stimulus input
        I_stim_exc = np.zeros((n_t,p['n_area']))

        area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
        
        # #==============TEST TEST TEST XXXXX DEBUG DEBUG=========================
        # area_stim_idx2 = p['areas'].index('9/46d') # Index of stimulated area
        
        # if PULSE_INPUT:
        #     time_idx = (t_plot>100) & (t_plot<=350) # original
        #     # time_idx = (t_plot>100) & (t_plot<=T-1)
        #     I_stim_exc[time_idx, area_stim_idx] = 41.187
        #     # I_stim_exc[time_idx, area_stim_idx2] = 41.187
        # else:
        #     for i in range(p['n_area']):
        #         I_stim_exc[:,i]=gaussian_noise(0,1e-5,n_t)        
        #     I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t) #2,0.5
        
        if PULSE_INPUT:
            time_idx = (t_plot>500) & (t_plot<=5500)
            I_stim_exc[time_idx, area_stim_idx] = 12

        if TOTAL_INPUT:
            for i in range(p['n_area']):
                I_stim_exc[:, i] += gaussian_noise(0,5,n_t)
        
        #---------------------------------------------------------------------------------
        # Storage
        #---------------------------------------------------------------------------------

        r_exc = np.zeros((n_t,p['n_area']))
        r_inh = np.zeros((n_t,p['n_area']))
        i_local_EE = np.zeros((n_t,p['n_area']))
        i_longrange_EE = np.zeros((n_t,p['n_area']))
        i_local_EI = np.zeros((n_t,p['n_area']))

        #---------------------------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------------------------
        fI = lambda x : x*(x>0)
        # fI = lambda x : x
        #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)  #for GATING_PATHWAY==1 only
        
        # Set activity to background firing
        r_exc[0] = r_exc_tgt
        r_inh[0] = r_inh_tgt
        
        #---------------------------------------------------------------------------------
        # Running the network
        #---------------------------------------------------------------------------------

        for i_t in range(1, n_t):
            longrange_E = np.dot(fln_scaled,r_exc[i_t-1])
            longrange_I = np.dot(fln_scaled_inh,r_exc[i_t-1])
            I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                     p['beta_exc'] * p['muEE'] * longrange_E +
                     I_bkg_exc + I_stim_exc[i_t])

            I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                     p['beta_inh'] * p['muIE'] * longrange_I + I_bkg_inh)
            
            # Record local currents
            i_local_EE[i_t] = local_EE*r_exc[i_t-1]
            i_local_EI[i_t] = local_EI*r_inh[i_t-1]
            i_longrange_EE[i_t] = p['beta_exc'] * p['muEE'] * longrange_E

            if GATING_PATHWAY:
                d_local_EI=np.zeros_like(local_EI)
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V4','8m']
                 
                for name in area_name_list:
                    area_idx=p['areas'].index(name)
                    d_local_EI[area_idx]=-local_EI[area_idx]*0.07  #0.1
                
                if I_stim_exc[i_t,area_stim_idx]>10:
                    I_exc=I_exc+d_local_EI*r_inh[i_t-1]
                
            d_r_exc = -r_exc[i_t-1] + fI(I_exc)
            d_r_inh = -r_inh[i_t-1] + fI(I_inh)

            r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
            r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']

        if plot_Flag:
            if CONSENSUS_CASE==0:
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
                    else:
                        area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
                else:
                    if VISUAL_INPUT:
                        area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
                    else:
                        area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
            else:
                if MACAQUE_CASE:
                    if VISUAL_INPUT:
                        area_name_list = ['V1','V4','8m 8l 8r','5','TEO TEOm','F4','9/46d 46d','TEpd TEa/ma TEa/mp','F7']
                    else:
                        raise SystemExit('Must give Visual input to networks under consensus map!')
                        
                else:
                    if VISUAL_INPUT:
                        area_name_list = ['V1','V2','V4','PEC PE','LIP','PGM','A32 A32V','A6DR','A6Va A6Vb']
                    else:
                        raise SystemExit('Must give Visual input to networks under consensus map!')

            area_idx_list=[-1]
            for name in area_name_list:
                area_idx_list=area_idx_list+[p['areas'].index(name)]
            #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
            
            f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(12, 12))
            
            clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
            c_color=0
            for ax, area_idx in zip(ax_list, area_idx_list):
                if area_idx < 0:
                    y_plot = I_stim_exc[:, area_stim_idx].copy()
                    z_plot = np.zeros_like(y_plot)
                    txt = 'Input'
    
                else:
                    y_plot = r_exc[:,area_idx].copy()
                    z_plot = r_inh[:,area_idx].copy()
                    txt = p['areas'][area_idx]
    
                if PULSE_INPUT:
                    y_plot = y_plot - y_plot.min()
                    # y_plot = y_plot - 10
                    z_plot = z_plot - z_plot.min()
                    ax.plot(t_plot, y_plot,color='k', linewidth=3)
                    #ax.plot(t_plot, z_plot,'--',color='b')
                else:
                    #ax.plot(t_plot, y_plot,color='r')
                    ax.plot(t_plot, y_plot - r_exc_base,color='r', linewidth=2)
                    # ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')
                    
                # ax.plot(t_plot, y_plot,color=clist[0][c_color])
                # ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
                c_color=c_color+1
                ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=18)
    
                if PULSE_INPUT:
                    ax.set_yticks([0,y_plot.max()])
                    # ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
                    ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max())], fontsize=18)
                ax.spines["right"].set_visible(False)
                ax.spines["top"].set_visible(False)
                #ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')
                plt.xticks(fontsize=18)

            f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=18)
            ax.set_xlabel('Time (ms)', fontsize=18)    
            
            if PULSE_INPUT:
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                plt.figure(figsize=(12, 8))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    # if area_idx < 0:
                    #     continue
                    # else:
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    plt.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    c_color=c_color+1
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                plt.xlabel('Time (ms)', fontsize=18)

                decay_time=np.zeros(p['n_area'])
                # fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,20))
                fig, ax=plt.subplots(figsize=(15,10))
                posi_array=np.arange(np.size(time_idx))
                get_posi=posi_array[time_idx]
                get_posi=get_posi[-1]
                t_plot_cut = t_plot[get_posi:-1].copy()-t_plot[get_posi]
                clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
                c_color=0
                for area_idx in np.arange(p['n_area']):
                    y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
                    y_plot_cut=y_plot_cut/y_plot_cut.max()
                    # ax1.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
                    p_end=np.where(y_plot_cut>1/np.e)[0][-1]
                    decay_time[c_color]=p_end*dt
                    c_color=c_color+1
                
                # ax1.set_xlabel('time (ms)')
                # ax1.set_ylabel('normalized response')
                
        
                ax.bar(np.arange(len(p['areas'])),decay_time,width = 1,color=clist[0])
                ax.set_xticks(np.arange(len(p['areas'])))
                ax.set_xticklabels(p['areas'],rotation=90, fontsize=16)
                ax.set_yscale('log')
                plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0, fontsize=20)
                ax.set_ylabel('Decay time (ms)',fontsize=20)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            else:
                decay_time=np.zeros(p['n_area']) 
            
            max_rate=np.max(r_exc-r_exc_base,axis=0)
            # network_graph_plot(p,max_rate,MACAQUE_CASE=MACAQUE_CASE)
            
            fig,ax=plt.subplots(figsize=(15,10))
            ax.plot(np.arange(len(p['areas'])), max_rate,'-o')
            ax.set_xticks(np.arange(len(p['areas'])))
            ax.set_xticklabels(p['areas'],rotation=90,fontsize=16)
            ax.set_yscale('log')
            ax.set_ylabel('Max Firing Rate',fontsize=16)
            # ax.set_xlabel('hierarchy values')

        if plot_Flag:
            return I_stim_exc, r_exc, r_inh, i_local_EE, i_local_EI, i_longrange_EE, I_bkg_exc, I_bkg_inh, area_stim_idx, dt, t_plot, f
        else:
            return I_stim_exc, r_exc, r_inh, i_local_EE, i_local_EI, i_longrange_EE, I_bkg_exc, I_bkg_inh, area_stim_idx, dt, t_plot


#run simulation of netowrk responses when there is a feedforward-feedback asymmetry
def run_stimulus_longrange_ei_asymmetry(p_t,VISUAL_INPUT=1,PULSE_INPUT=1,MACAQUE_CASE=1):
        
        if VISUAL_INPUT:
            area_act = 'V1'
        else:
            if MACAQUE_CASE:
                area_act='2'
            else:
                area_act = 'AuA1'
        print('Running network in the case of longrange EI asymmetry with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

        #---------------------------------------------------------------------------------
        # Redefine Parameters
        #---------------------------------------------------------------------------------

        p=p_t.copy()

        # Definition of combined parameters

        local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
        local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
        local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
        local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)
        
        fln_scaled_EE = (p['exc_scale'] * p['fln_mat'].T * p['sln_mat'].T).T
        fln_scaled_IE = (p['exc_scale'] * p['fln_mat'].T * (1-p['sln_mat']).T).T    
            
        #---------------------------------------------------------------------------------
        # Simulation Parameters
        #---------------------------------------------------------------------------------

        dt = 0.5   # ms
        if PULSE_INPUT:
            T = 2500 
        else:                
            T = 1e5
        
        t_plot = np.linspace(0, T, int(T/dt)+1)
        n_t = len(t_plot)  

        # From target background firing inverts background inputs
        r_exc_tgt = 10 * np.ones(p['n_area'])    
        r_inh_tgt = 35 * np.ones(p['n_area'])

        longrange_EE = np.dot(fln_scaled_EE,r_exc_tgt)
        longrange_IE = np.dot(fln_scaled_IE,r_exc_tgt)
        
        I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                                 + p['beta_exc']*p['muEE']*longrange_EE)
        I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                                 + p['beta_inh']*p['muIE']*longrange_IE)

        # Set stimulus input
        I_stim_exc = np.zeros((n_t,p['n_area']))

        area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
        
        
        if PULSE_INPUT:
            time_idx = (t_plot>100) & (t_plot<=350)
            I_stim_exc[time_idx, area_stim_idx] = 41.187
        else:
            for i in range(p['n_area']):
                I_stim_exc[:,i]=gaussian_noise(0,1e-5,n_t)        
            I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t)
        
        #---------------------------------------------------------------------------------
        # Storage
        #---------------------------------------------------------------------------------

        r_exc = np.zeros((n_t,p['n_area']))
        r_inh = np.zeros((n_t,p['n_area']))

        #---------------------------------------------------------------------------------
        # Initialization
        #---------------------------------------------------------------------------------
        fI = lambda x : x*(x>0)
        #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)  #for GATING_PATHWAY==1 only
        
        # Set activity to background firing
        r_exc[0] = r_exc_tgt
        r_inh[0] = r_inh_tgt
        
        #---------------------------------------------------------------------------------
        # Running the network
        #---------------------------------------------------------------------------------

        for i_t in range(1, n_t):
            longrange_EE = np.dot(fln_scaled_EE,r_exc[i_t-1])
            longrange_IE = np.dot(fln_scaled_IE,r_exc[i_t-1])
            
            I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                     p['beta_exc'] * p['muEE'] * longrange_EE +
                     I_bkg_exc + I_stim_exc[i_t])

            I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                     p['beta_inh'] * p['muIE'] * longrange_IE + I_bkg_inh)
            
                
            d_r_exc = -r_exc[i_t-1] + fI(I_exc)
            d_r_inh = -r_inh[i_t-1] + fI(I_inh)

            r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
            r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']
        
        #---------------------------------------------------------------------------------
        # Plotting step input results
        #---------------------------------------------------------------------------------
        
        if MACAQUE_CASE:
            if VISUAL_INPUT:
                area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
            else:
                area_name_list = ['5','2','F1','10','9/46v','9/46d','F5','7B','F2','ProM','F7','8B','24c']
        else:
            if VISUAL_INPUT:
                area_name_list = ['V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
            else:
                area_name_list = ['AuA1','A1-2','V1','V2','V4','MST','LIP','PG','A23a','A6DR','A6Va']
            

            
        area_idx_list=[-1]
        for name in area_name_list:
            area_idx_list=area_idx_list+[p['areas'].index(name)]
        #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
        
        f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(16, 16))
        
        clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
        c_color=0
        for ax, area_idx in zip(ax_list, area_idx_list):
            if area_idx < 0:
                y_plot = I_stim_exc[:, area_stim_idx].copy()
                z_plot = np.zeros_like(y_plot)
                txt = 'Input'

            else:
                y_plot = r_exc[:,area_idx].copy()
                z_plot = r_inh[:,area_idx].copy()
                txt = p['areas'][area_idx]

   
            y_plot = y_plot - y_plot.min()
            z_plot = z_plot - z_plot.min()
            ax.plot(t_plot, y_plot,color=clist[0][c_color])
            ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
            c_color=c_color+1
            ax.text(0.9, 0.6, txt, transform=ax.transAxes)

            ax.set_yticks([0,y_plot.max(),z_plot.max()])
            ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            #ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
        ax.set_xlabel('Time (ms)')    
        
        max_rate=np.max(r_exc)
        #network_graph_plot(p,max_rate,MACAQUE_CASE=MACAQUE_CASE)
        
        return I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot
 

#simulate the network by giving white noise input and estimate the time constant by fitting the autocorrelation function
def plt_white_noise_input(p_t,VISUAL_INPUT=1,MACAQUE_CASE=1,NO_gradient=0,NO_feedback=0,NO_long_link=0, T=5e6, TOTAL_INPUT=1):
    
    print('In function: plt_white_noise_input, parameters are VISUAL_INPUT='+str(VISUAL_INPUT)+'  MACAQUE_CASE='+str(MACAQUE_CASE)+'  NO_gradient='+str(NO_gradient)+'  NO_feedback='+str(NO_feedback)+'  NO_long_link='+str(NO_long_link))
    
    if NO_gradient+NO_feedback+NO_long_link>2:
        raise SystemExit('Conflict of network parameter setting!')
        
    if NO_gradient+NO_feedback+NO_long_link==1:
        NORMAL_case=0
    else:
        NORMAL_case=1
            
    p = p_t.copy()
    I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate \
    =run_stimulus(p,VISUAL_INPUT=VISUAL_INPUT,PULSE_INPUT=0,MACAQUE_CASE=MACAQUE_CASE,TOTAL_INPUT=TOTAL_INPUT, T=T, plot_Flag=0)
    
    # #area_name_list = ['V1','V4','8m','8l','TEO','2','7A','10','9/46v','9/46d','TEpd','7m','7B','24c']
    # area_name_list=p['areas']
    # area_idx_list=[-1]
    
    # T_lag=int(3e3)
    # acf_data=np.zeros((len(area_name_list),T_lag+1))   
    
    # for name in area_name_list:
    #         area_idx_list=area_idx_list+[p['areas'].index(name)]
            
    # f, ax_list = plt.subplots(len(area_idx_list), sharex=True,figsize=(10,15))
    # j=0
    # for ax, area_idx in zip(ax_list, area_idx_list):
    #     if area_idx < 0:
    #         y_plot = I_stim_exc[::int(1/dt), area_stim_idx].copy()
    #         txt = 'Input'
            
    #     else:
    #         y_plot = r_exc[::int(1/dt),area_idx].copy()
    #         txt = p['areas'][area_idx]
    #         acf_data[j,:] = smt.stattools.acf(y_plot,nlags=T_lag, fft=True)
    #         j=j+1
            
    #     y_plot = y_plot - y_plot.min()
    #     ax.plot(t_plot[::int(1/dt)], y_plot)
    #     ax.text(0.9, 0.6, txt, transform=ax.transAxes)

    #     ax.set_yticks([y_plot.max()])
    #     ax.set_yticklabels(['{:0.4f}'.format(y_plot.max())])
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     ax.yaxis.set_ticks_position('left')

    # f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
    # ax.set_xlabel('Time (ms)')
                
    # clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]

    # for name in area_name_list:
    #     area_idx_list=area_idx_list+[p['areas'].index(name)]
    
    # fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,20))
    # for i in np.arange(len(area_name_list)):
    #     ax1.plot(np.arange(T_lag+1),acf_data[i,:],color=clist[0][i])
        
    # ax1.set_xlabel('Time difference (ms)', fontsize=16)
    # ax1.set_ylabel('Correlation', fontsize=16)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # #plt.savefig('result/correlation_stim_V1.pdf')
    
     
    # #---------------------------------------------------------------------------------
    # # parameter fit
    # #---------------------------------------------------------------------------------    
    # t_plot=t_plot[::int(1/dt)].copy()
     
    # delay_time=np.zeros(len(area_name_list))
    # f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
    # for ax, i in zip(ax_list, np.arange(len(area_name_list))):
    #     p_end=np.where(acf_data[i,:]>0.05)[0][-1]
        
    #     r_single, _ =optimize.curve_fit(single_exp,t_plot[0:p_end],acf_data[i,0:p_end])
    #     r_double, _ =optimize.curve_fit(double_exp,t_plot[0:p_end],acf_data[i,0:p_end],p0=[r_single[0],0.1,r_single[1],0],bounds=(0,np.inf),maxfev=5000)
        
    #     e_single=sum((acf_data[i,0:p_end]-r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]))**2)
    #     e_double=sum((acf_data[i,0:p_end]-(r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1])))**2)
        
    #     e_ratio=e_single/e_double
        
    #     if e_ratio>8:
    #         delay_time[i]=r_double[0]
    #     else:
    #         delay_time[i]=r_single[0]
                
    #     #print('error ratio of',area_name_list[i],"=",str(e_ratio))
        
    #     ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
    #     ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
    #     ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
    #     ax.set_ylim(0,1)
    #     txt = area_name_list[i]
    #     ax.text(0.9, 0.6, txt, transform=ax.transAxes)
        
    # f.text(0.01, 0.5, 'Simulated correlation', va='center', rotation='vertical')
    # ax.set_xlabel('Time difference (ms)')
  
    # ax2.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
    # ax2.set_xticks(np.arange(len(area_name_list)))
    # ax2.set_xticklabels(area_name_list,rotation=90)
    # ax2.set_yticks([10,100,1000])
    # ax2.set_yticklabels(['10 ms','100 ms','1 s'],rotation=0)
    # ax2.set_ylabel('$Simulated T_{delay}$ (ms)')
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    
    # ind=np.argsort(delay_time)
    # delay_time_sort=np.zeros_like(delay_time)
    # area_name_list_sort=[]
    # for i in np.arange(len(ind)):
    #     delay_time_sort[i]=delay_time[ind[i]]
    #     area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
        
    # plt.figure(figsize=(5,7))        
    # ax = plt.axes()
    # plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
    # plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
    # plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    # plt.ylabel('$Simulated T_{delay}$ (ms)')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    
    # if NO_gradient:
    #     plt.savefig('result/NO_gradient_hist.pdf')
    # if NO_feedback:
    #     plt.savefig('result/NO_feedback_hist.pdf')
    # if NO_long_link:
    #     plt.savefig('result/NO_long_link_hist.pdf')
    # if NORMAL_case:
    #     plt.savefig('result/normal.pdf')
    return r_exc
        
    
#compute how the eigenvalues and eigenvectors change as long-range strength increases      
def perturbation_analysis_connectivity_matrix(p_t,MACAQUE_CASE=1,CONSENSUS_CASE=0):
    
    p=p_t.copy()
    
    _,W0=genetate_net_connectivity(p,ZERO_FLN=1,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    p,W1=genetate_net_connectivity(p,ZERO_FLN=0,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
       
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    A=W0_EI
    B=W1_EI-W0_EI
    n_steps = 100
    n=2*p['n_area']
    
    egval, egvec = np.linalg.eig(A)
    egvec=egvec+0j
    
    id_sort = np.argsort(abs(egval.real))
    vec_dir_ref = (egval[id_sort], egvec[:, id_sort])
    
    s_vecs = np.zeros((n_steps, n, n), dtype='complex128')
    s_vals = np.zeros((n_steps, n), dtype='complex128')
    
    app_vecs_1 = np.zeros((n_steps, n, n), dtype='complex128')
    app_vecs_2 = np.zeros((n_steps, n, n), dtype='complex128')
    
    app_vals_1 = np.zeros((n_steps, n), dtype='complex128')
    app_vals_2 = np.zeros((n_steps, n), dtype='complex128')
    
    vec_error_1=np.zeros((n_steps, n), dtype='complex128')
    vec_error_2=np.zeros((n_steps, n), dtype='complex128')
    
    for k in range(n_steps):
        h = A + 0.99*k/n_steps * B
        egval, egvec = np.linalg.eig(h)
        #eig_pick, vec_dir_ref = egval, egvec
        vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
        s_vals[k, :] = vec_dir_ref[0]
        s_vecs[k, :, :] = vec_dir_ref[1]
        
        app_eigmode=eig_approx(A,h-A)
        app_vals_1[k,:]=app_eigmode[0]
        app_vals_2[k,:]=app_eigmode[1]
        app_vecs_1[k,:,:]=app_eigmode[2]
        app_vecs_2[k,:,:]=app_eigmode[3]
        
        for s in range(n):
            vec_error_1[k,s]=np.linalg.norm(np.abs(s_vecs[k,:,s])-np.abs(app_vecs_1[k,:,s]))
            vec_error_2[k,s]=np.linalg.norm(np.abs(s_vecs[k,:,s])-np.abs(app_vecs_2[k,:,s]))
        
    tt = linspace(0, 1, n_steps)
    
    # plt.figure(111)
    # plt.plot(tt,vec_error_1,'-')
    #plt.plot(tt,vec_error_2,'--')
    
    plt.figure()
    plt.plot(np.real(s_vals[0,:]),np.real(s_vals[-1,:]),'ro')
    plt.plot(np.real(s_vals[0,:]),np.real(app_vals_1[-1,:]),'b*')
    plt.plot(linspace(-0.1,0,100),linspace(-0.1,0,100),'-k')
    plt.xlabel('eigval-real without long-range connection')
    plt.ylabel('eigval-real with long-range connection')
    
    # plt.figure(34)
    # plt.plot(tt,s_vals.real,label='true')
    # plt.plot(tt,app_vals_1.real,'--',label='first order')
    # plt.plot(tt,app_vals_2.real,'-.',label='second order')
    # plt.ylabel('eigval-real')
    # plt.xlabel('noise level')
    # plt.legend()
    
    plt.figure()
    plt.plot(tt,s_vals.real,label='true')
    plt.ylabel('eigval-real')
    plt.xlabel('noise level')
    
    # plt.figure(35)
    # plt.plot(tt,s_vals.imag,label='true')
    # plt.plot(tt,app_vals_1.imag,'--',label='first order')
    # plt.plot(tt,app_vals_2.imag,'-.',label='second order')
    # plt.ylabel('eigval-imag')
    # plt.xlabel('noise level')
    # plt.legend()
    
    plt.figure()
    plt.plot(tt,s_vals.imag,label='true')
    plt.ylabel('eigval-imag')
    plt.xlabel('noise level')
    
    # fig = plt.figure(234)
    # ax = fig.gca(projection='3d')
    # for ie in range(n):
    #   ax.plot3D(tt, s_vals.real[:,ie], s_vals.imag[:,ie])
    
    # ax.set_xlabel('t')
    # ax.set_ylabel('eigval-real')
    # ax.set_zlabel('eigval-imag')
    
    
    # plt.figure(36)
    # for k in range(n):
    #     plt.plot(s_vals.real[:,k],s_vals.imag[:,k],alpha=0.3)
    # plt.plot(s_vals.real[0,:],s_vals.imag[0,:],'o')
    # plt.plot(egval.real,egval.imag,'o')
    
    # plt.xlabel('eigval-real')
    # plt.ylabel('eigval-imag')
    # plt.show()


    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(app_vecs_1[-50,:,:]),vmin = 0, vmax = 1,cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization-approximation')

    x = np.arange(2*p['n_area']) # xticks
    xlim = (0,2*p['n_area'])
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(np.real(app_vals_1[-1,:]))
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(s_vecs[-50,:,:]),vmin = 0, vmax = 1,cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('eigenvector visualization-approximation')
      
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_xticks(x[::1])
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    ax.set_xticklabels(np.real(s_vals[-1,:]))
    plt.setp(ax.get_xticklabels(), rotation=90)
        

    
#fitting the time constants of each area from its theoretical autocorrelation function when giving an input at a specific input location     
def theoretical_time_constant_input_at_one_area(p_t,eigVecs,eigVals,input_loc='V1'):
    p=p_t.copy()
    area_name_list=p['areas']
    
    inv_eigVecs=np.linalg.inv(eigVecs)
    
    n=len(area_name_list)
    T_lag=int(3e3)
    
    m=p['areas'].index(input_loc)         
    acf_data=np.zeros((n,T_lag+1))+0j 
    coef=np.zeros(2*n)+0j

    for i in np.arange(n):
        for s in np.arange(T_lag+1):
            for j in np.arange(2*n):
                coef[j]=0
                for k in np.arange(2*n):
                    coef[j]=coef[j]+eigVecs[i,j]*eigVecs[i,k]*inv_eigVecs[j,m]*inv_eigVecs[k,m]/(-eigVals[j]-eigVals[k])
                acf_data[i,s] = acf_data[i,s]+coef[j]*np.exp(eigVals[j]*s)
        acf_data[i,:]=acf_data[i,:]/acf_data[i,0]
            
    clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]
    
    plt.figure(figsize=(10,5))
    ax = plt.axes()
    for i in np.arange(len(area_name_list)):
        plt.plot(np.arange(T_lag+1),acf_data[i,:],color=clist[0][i])
        
    plt.legend(area_name_list)
    plt.xlabel('Time difference (ms)')
    plt.ylabel('Theoretical correlation')
    plt.title('input_loc='+input_loc)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #plt.savefig('result/correlation_stim_V1.pdf')
    
    t_plot=np.arange(T_lag)
     
    delay_time=np.zeros(len(area_name_list))
    f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
    for ax, i in zip(ax_list, np.arange(len(area_name_list))):
        p_end=np.where(acf_data[i,:]>0.05)[0][-1]
        
        r_single, _ =optimize.curve_fit(single_exp,t_plot[0:p_end],acf_data[i,0:p_end])
        r_double, _ =optimize.curve_fit(double_exp,t_plot[0:p_end],acf_data[i,0:p_end],p0=[r_single[0],0.1,r_single[1],0],bounds=(0,np.inf),maxfev=5000)
        
        e_single=sum((acf_data[i,0:p_end]-r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]))**2)
        e_double=sum((acf_data[i,0:p_end]-(r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1])))**2)
        
        e_ratio=e_single/e_double
        
        if e_ratio>8:
            delay_time[i]=r_double[0]
        else:
            delay_time[i]=r_single[0]
                
        #print('error ratio of',area_name_list[i],"=",str(e_ratio))
        
        ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
        ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
        ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
        ax.set_ylim(0,1)
        txt = area_name_list[i]
        ax.text(0.9, 0.6, txt, transform=ax.transAxes)
        
    f.text(0.01, 0.5, 'Theoretical Correlation', va='center', rotation='vertical')
    ax.set_xlabel('Time difference (ms)')
    ax.set_title('input_loc='+input_loc)
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list,rotation=90)
    #plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc='+input_loc)
    
    ind=np.argsort(delay_time)
    delay_time_sort=np.zeros_like(delay_time)
    area_name_list_sort=[]
    for i in np.arange(len(ind)):
        delay_time_sort[i]=delay_time[ind[i]]
        area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
        
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
    #plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc='+input_loc)
    
#compute the time constants of each area by Green's function method    
def theoretical_time_constant_input_at_all_areas(p_t,eigVecs,eigVals):
    p=p_t.copy()
    area_name_list=p['areas']
    
    print('inv_cond=',np.linalg.cond(eigVecs))
    inv_eigVecs=np.linalg.inv(eigVecs)
    
    n=len(area_name_list)
    T_lag=int(1e3)
            
    acf_data=np.zeros((n,T_lag+1))+0j 
    coef=np.zeros((n,2*n))+0j
    coef_green=np.zeros((n,2*n))+0j  #cofficient of the green's function 
    #m=0  #9 for area 2
    for i in np.arange(n):
        m=i
        for s in np.arange(T_lag+1):
            for j in np.arange(2*n):
                coef_green[i,j]=eigVecs[i,j]*inv_eigVecs[j,m]
                coef[i,j]=0
                for k in np.arange(2*n):
                    coef[i,j]=coef[i,j]+eigVecs[i,j]*eigVecs[i,k]*inv_eigVecs[j,m]*inv_eigVecs[k,m]/(-eigVals[j]-eigVals[k])
                acf_data[i,s] = acf_data[i,s]+coef[i,j]*np.exp(eigVals[j]*s)
        acf_data[i,:]=acf_data[i,:]/acf_data[i,0]
             
    clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]
    
    plt.figure(figsize=(10,5))
    ax = plt.axes()
    for i in np.arange(len(area_name_list)):
        plt.plot(np.arange(T_lag+1),acf_data[i,:],color=clist[0][i])
        
    plt.legend(area_name_list)
    plt.xlabel('Time difference (ms)')
    plt.ylabel(' Theoretical correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend('input loc and corr measure at the same area')
    #plt.savefig('result/correlation_stim_V1.pdf')
    
    t_plot=np.arange(T_lag)
     
    delay_time=np.zeros(len(area_name_list))
    f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
    for ax, i in zip(ax_list, np.arange(len(area_name_list))):
        p_end=np.where(acf_data[i,:]>0.05)[0][-1]
        
        r_single, _ =optimize.curve_fit(single_exp,t_plot[0:p_end],acf_data[i,0:p_end])
        r_double, _ =optimize.curve_fit(double_exp,t_plot[0:p_end],acf_data[i,0:p_end],p0=[r_single[0],0.1,r_single[1],0],bounds=(0,np.inf),maxfev=5000)
        
        e_single=sum((acf_data[i,0:p_end]-r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]))**2)
        e_double=sum((acf_data[i,0:p_end]-(r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1])))**2)
        
        e_ratio=e_single/e_double
        
        if e_ratio>8:
            delay_time[i]=r_double[0]
        else:
            delay_time[i]=r_single[0]
                
        #print('error ratio of',area_name_list[i],"=",str(e_ratio))
        
        ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
        ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
        ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
        ax.set_ylim(0,1)
        txt = area_name_list[i]
        ax.text(0.9, 0.6, txt, transform=ax.transAxes)
        
    f.text(0.01, 0.5, 'Theoretical Correlation', va='center', rotation='vertical')
    ax.set_xlabel('Time difference (ms)')
    ax.set_title('input_loc and corr measure at the same area')
    
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list,rotation=90)
    plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc and corr measure at the same area')
    
    ind=np.argsort(delay_time)
    delay_time_sort=np.zeros_like(delay_time)
    area_name_list_sort=[]
    for i in np.arange(len(ind)):
        delay_time_sort[i]=delay_time[ind[i]]
        area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
        
    plt.figure(figsize=(5,7))        
    ax = plt.axes()
    plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
    plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
    plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
    plt.ylabel('Theoretical $T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('input_loc and corr measure at the same area')
    
    tau_s=np.zeros(2*n)
    for i in range(len(tau_s)):
        tau_s[i]=float(format(np.real(-1/eigVals[i]),'.2f'))
    
    ind=np.argsort(-tau_s)
    coef_reorder=np.zeros((p['n_area'],2*p['n_area']))+0j
    coef_green_reorder=np.zeros((p['n_area'],2*p['n_area']))+0j
    tau_reorder=np.zeros(2*p['n_area'])
    
    for i in range(2*p['n_area']):
        coef_reorder[:,i]=coef[:,ind[i]]
        coef_green_reorder[:,i]=coef_green[:,ind[i]]
        tau_reorder[i]=tau_s[ind[i]]
    
    coef_normed=np.zeros_like(coef)
    coef_normed_green=np.zeros_like(coef_green)

    #normalize the coefficient row by row
    for j in range(p['n_area']):
        coef_normed[j,:]=coef_reorder[j,:]/np.max(np.abs(coef_reorder[j,:]))
        coef_normed_green[j,:]=coef_green_reorder[j,:]/np.max(np.abs(coef_green_reorder[j,:]))
        
    fig, ax = plt.subplots(figsize=(20,10))
    f=ax.pcolormesh(np.abs(coef_normed),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('full coef matrix of autocorrelation')
        
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(tau_reorder)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')    
    
        
    fig, ax = plt.subplots(figsize=(12,10))
    f=ax.pcolormesh(np.abs(coef_normed[:,:p['n_area']]),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('E population coef matrix of autocorrelation')
        
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(tau_reorder[:p['n_area']])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')    
    
    
    fig, ax = plt.subplots(figsize=(12,10))
    f=ax.pcolormesh(np.abs(coef_normed_green[:,:p['n_area']]),cmap='hot')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('E population coef matrix of Greens func')
        
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::1])
    ax.invert_xaxis()
        
    ax.set_xticklabels(tau_reorder[:p['n_area']])
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    #fig.savefig('result/full_eigenmode.pdf')    
    
#see how eigenvalue changes by gradually changing the strength of all long-range connections or gradually adding long-range connection from small strength to large strength 
def role_of_connection_by_adding_long_range_connection(p_t,MACAQUE_CASE=1,LINEAR_HIER=0,CONNECTION_CASE=0,CONSENSUS_CASE=0):
    #CONNECTION_CASE=0 gradually change the strength of all long-range connections
    #CONNECTION_CASE=1 gradually add long-range connection from small strength to large strength
    p=p_t.copy()
    
    _,W0=genetate_net_connectivity(p,ZERO_FLN=1,LINEAR_HIER=LINEAR_HIER,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    p,W1=genetate_net_connectivity(p,ZERO_FLN=0,LINEAR_HIER=LINEAR_HIER,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
       
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    A=W0_EI
    n=2*p['n_area']
    B=W1_EI-W0_EI
    B_EE=B[:p['n_area'],:p['n_area']]
    B_IE=B[p['n_area']:,:p['n_area']]
    B_EE_flat=B_EE.flatten()
    B_EE_flat=B_EE_flat[B_EE_flat>0]
    B_IE_flat=B_IE.flatten()
    B_IE_flat=B_IE_flat[B_IE_flat>0]
    idx_EE=np.argsort(B_EE_flat)
    idx_IE=np.argsort(B_IE_flat)
    
    print('mean long-range strength=',np.mean(B_EE_flat))
    
    if CONNECTION_CASE==0:
        n_steps = 100
    else:  
        n_steps = int(sum(sum(B_EE>0))) #number of nozero long-range projections
        plt.figure()
        plt.plot(np.arange(n_steps),np.sort(B_EE_flat),label='true')
        plt.ylabel('EE FLN strength')
        plt.xlabel('index')
        plt.yscale('log')
    
    egval, egvec = np.linalg.eig(A)
    egvec=egvec+0j
    
    id_sort = np.argsort(abs(egval.real))
    vec_dir_ref = (egval[id_sort], egvec[:, id_sort])
    
    s_vecs = np.zeros((n_steps, n, n), dtype='complex128')
    s_vals = np.zeros((n_steps, n), dtype='complex128')
    
    app_vecs_1 = np.zeros((n_steps, n, n), dtype='complex128')
    app_vecs_2 = np.zeros((n_steps, n, n), dtype='complex128')
    
    app_vals_1 = np.zeros((n_steps, n), dtype='complex128')
    app_vals_2 = np.zeros((n_steps, n), dtype='complex128')
    
    vec_error_1=np.zeros((n_steps, n), dtype='complex128')
    vec_error_2=np.zeros((n_steps, n), dtype='complex128')
    
    for k in range(n_steps):
        if CONNECTION_CASE==0:
            h = A + k/n_steps * B
        if CONNECTION_CASE==1:
            temp_B_EE=B_EE.copy()
            temp_B_IE=B_IE.copy()
            temp_B=B.copy()
            temp_B_EE[temp_B_EE>B_EE_flat[idx_EE[k]]]=0
            temp_B_IE[temp_B_IE>B_IE_flat[idx_IE[k]]]=0
            temp_B[:p['n_area'],:p['n_area']]=temp_B_EE
            temp_B[p['n_area']:,:p['n_area']]=temp_B_IE
            h=A+temp_B
            
        egval, egvec = np.linalg.eig(h)
        #eig_pick, vec_dir_ref = egval, egvec
        vec_dir_ref = pick_eigen_direction(egval, egvec, vec_dir_ref, 'close-egval')
        s_vals[k, :] = vec_dir_ref[0]
        s_vecs[k, :, :] = vec_dir_ref[1]
        
        app_eigmode=eig_approx(A,h-A)
        app_vals_1[k,:]=app_eigmode[0]
        app_vals_2[k,:]=app_eigmode[1]
        app_vecs_1[k,:,:]=app_eigmode[2]
        app_vecs_2[k,:,:]=app_eigmode[3]
        
        for s in range(n):
            vec_error_1[k,s]=np.linalg.norm(np.abs(s_vecs[k,:,s])-np.abs(app_vecs_1[k,:,s]))
            vec_error_2[k,s]=np.linalg.norm(np.abs(s_vecs[k,:,s])-np.abs(app_vecs_2[k,:,s]))
        
    tt = linspace(0, 1, n_steps)
    
    plt.figure()
    plt.plot(np.real(-1/s_vals[0,:p['n_area']]),np.real(-1/s_vals[-1,:p['n_area']]),'ro')
    plt.plot(np.real(-1/s_vals[0,:p['n_area']]),np.real(-1/app_vals_2[-1,:p['n_area']]),'b*')
    plt.plot(linspace(-0.1,0,100),linspace(-0.1,0,100),'-k')
    plt.xlabel('time constants without long-range connection')
    plt.ylabel('time constants with long-range connection')
    
    plt.figure()
    plt.plot(tt,-1/s_vals[:,:p['n_area']].real,label='true')
    plt.ylabel('time constants')
    plt.xlabel('long range strength level')
    
        
    strength_level=1
    posi=round(strength_level*(n_steps-1))
#==================================full matrix=================================

    fig, ax = plt.subplots(2,2,figsize=(20,20))
    f=ax[0,0].pcolormesh(np.abs(s_vecs[0,:,:]),cmap='hot')
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    ax[0,0].set_title('eigenvector visualization-no long range')

    tau=-1/np.real(s_vals[0,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
    
    x = np.arange(2*p['n_area']) # xticks
    y = np.arange(2*p['n_area']) # yticks
    xlim = (0,2*p['n_area'])
    ylim = (0,2*p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]+p['areas'][::2]
    yticklabels_even=p['areas'][::2]+p['areas'][1::2]
    
    # set original ticks and ticklabels
    ax[0,0].set_xlim(xlim)
    ax[0,0].set_ylim(ylim)
    ax[0,0].set_xticks(x[::1])
    ax[0,0].invert_xaxis()
        
    ax[0,0].set_xticklabels(tau_s)
    ax[0,0].set_yticks(y[::2])
    ax[0,0].set_yticklabels(yticklabels_even)
    ax[0,0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0,0].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0,0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    f=ax[0,1].pcolormesh(np.abs(s_vecs[posi,:,:]),cmap='hot')
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    ax[0,1].set_title('eigenvector visualization-with long range')

    
    tau=-1/np.real(s_vals[posi,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[0,1].set_xlim(xlim)
    ax[0,1].set_ylim(ylim)
    ax[0,1].set_xticks(x[::1])
    ax[0,1].invert_xaxis()
        
    ax[0,1].set_xticklabels(tau_s)
    ax[0,1].set_yticks(y[::2])
    ax[0,1].set_yticklabels(yticklabels_even)
    ax[0,1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0,1].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0,1].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   

    
    f=ax[1,0].pcolormesh(np.abs(app_vecs_1[posi,:,:]),vmin=0,vmax=1,cmap='hot') #norm=LogNorm(vmin=None,vmax=None)
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    ax[1,0].set_title('eigenvector visualization-with long-range approximation')
     
    
    tau=-1/np.real(app_vals_2[posi,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_xticks(x[::1])
    ax[1,0].invert_xaxis()
        
    ax[1,0].set_xticklabels(tau_s)
    ax[1,0].set_yticks(y[::2])
    ax[1,0].set_yticklabels(yticklabels_even)
    ax[1,0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1,0].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[1,0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    f=ax[1,1].pcolormesh(B,cmap='bwr')
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    ax[1,1].set_title('long-range connection')
   
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[1,1].set_xlim(xlim)
    ax[1,1].set_ylim(ylim)
    ax[1,1].set_xticks(x[::4])
    ax[1,1].set_xticklabels(xticklabels_even)
    ax[1,1].set_yticks(y[::4])
    ax[1,1].set_yticklabels(yticklabels_even)
    ax[1,1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1,1].get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax[1,1].twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[2::4])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[1,1].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[2::4])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
#==================================EE matrix===================================

    fig, ax = plt.subplots(2,2,figsize=(20,20))
    f=ax[0,0].pcolormesh(np.abs(s_vecs[0,:p['n_area'],:p['n_area']]),cmap='hot')
    fig.colorbar(f,ax=ax[0,0],pad=0.1)
    ax[0,0].set_title('eigenvector visualization-no long range')

    tau=-1/np.real(s_vals[0,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))

    
    x = np.arange(p['n_area']) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,p['n_area'])
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax[0,0].set_xlim(xlim)
    ax[0,0].set_ylim(ylim)
    ax[0,0].set_xticks(x[::1])
    ax[0,0].invert_xaxis()
        
    ax[0,0].set_xticklabels(tau_s[:p['n_area']])
    ax[0,0].set_yticks(y[::2])
    ax[0,0].set_yticklabels(yticklabels_even)
    ax[0,0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0,0].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0,0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    
    f=ax[0,1].pcolormesh(np.abs(s_vecs[posi,:p['n_area'],:p['n_area']]),cmap='hot')
    fig.colorbar(f,ax=ax[0,1],pad=0.1)
    ax[0,1].set_title('eigenvector visualization-with long range')

    
    tau=-1/np.real(s_vals[posi,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[0,1].set_xlim(xlim)
    ax[0,1].set_ylim(ylim)
    ax[0,1].set_xticks(x[::1])
    ax[0,1].invert_xaxis()
        
    ax[0,1].set_xticklabels(tau_s[:p['n_area']])
    ax[0,1].set_yticks(y[::2])
    ax[0,1].set_yticklabels(yticklabels_even)
    ax[0,1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0,1].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0,1].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   

    
    f=ax[1,0].pcolormesh(np.abs(app_vecs_1[posi,:p['n_area'],:p['n_area']]),vmin=0,vmax=0.7,cmap='hot') #norm=LogNorm(vmin=None,vmax=None)
    fig.colorbar(f,ax=ax[1,0],pad=0.1)
    ax[1,0].set_title('eigenvector visualization-with long-range approximation')
     
    
    tau=-1/np.real(app_vals_2[-1,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[1,0].set_xlim(xlim)
    ax[1,0].set_ylim(ylim)
    ax[1,0].set_xticks(x[::1])
    ax[1,0].invert_xaxis()
        
    ax[1,0].set_xticklabels(tau_s[:p['n_area']])
    ax[1,0].set_yticks(y[::2])
    ax[1,0].set_yticklabels(yticklabels_even)
    ax[1,0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1,0].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[1,0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    f=ax[1,1].pcolormesh(B[:p['n_area'],:p['n_area']],cmap='hot') #norm=LogNorm(vmin=None,vmax=None)
    fig.colorbar(f,ax=ax[1,1],pad=0.1)
    ax[1,1].set_title('long-range connection')
   
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    # set original ticks and ticklabels
    ax[1,1].set_xlim(xlim)
    ax[1,1].set_ylim(ylim)
    ax[1,1].set_xticks(x[::2])
    ax[1,1].set_xticklabels(xticklabels_even)
    ax[1,1].set_yticks(y[::2])
    ax[1,1].set_yticklabels(yticklabels_even)
    ax[1,1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1,1].get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax[1,1].twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[1,1].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    
    #===============================================================================
    fig, ax = plt.subplots(1,3,figsize=(20,20))
    
    bin_vec=np.abs(s_vecs[-1,:p['n_area'],:p['n_area']])
    bin_vec[np.where(bin_vec>0.1)]=1
    bin_vec[np.where(bin_vec<0.1)]=0
    
    f=ax[0].pcolormesh(bin_vec,cmap='hot')
    fig.colorbar(f,ax=ax[0],pad=0.1)
    ax[0].set_title('eigenvector visualization-with long range connection')

    
    tau=-1/np.real(s_vals[-1,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_xticks(x[::1])
    ax[0].invert_xaxis()
        
    ax[0].set_xticklabels(tau_s[:p['n_area']])
    ax[0].set_yticks(y[::2])
    ax[0].set_yticklabels(yticklabels_even)
    ax[0].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[0].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[0].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   

    
    bin_vec_app=np.abs(app_vecs_1[-1,:p['n_area'],:p['n_area']])
    bin_vec_app[np.where(bin_vec_app>0.07)]=1
    bin_vec_app[np.where(bin_vec_app<0.07)]=0
    
    f=ax[1].pcolormesh(bin_vec_app,cmap='hot') #norm=LogNorm(vmin=None,vmax=None)
    fig.colorbar(f,ax=ax[1],pad=0.1)
    ax[1].set_title('eigenvector visualization-with long-range approximation')
     
    
    tau=-1/np.real(app_vals_2[-1,:])
    tau_s=np.zeros_like(tau)
    
    for i in range(len(tau)):
        tau_s[i]=float(format(tau[i],'.2f'))
        
    # set original ticks and ticklabels
    ax[1].set_xlim(xlim)
    ax[1].set_ylim(ylim)
    ax[1].set_xticks(x[::1])
    ax[1].invert_xaxis()
        
    ax[1].set_xticklabels(tau_s[:p['n_area']])
    ax[1].set_yticks(y[::2])
    ax[1].set_yticklabels(yticklabels_even)
    ax[1].invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax[1].get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax[1].twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()  
    
    f=ax[2].plot(np.abs(s_vecs[-1,:p['n_area'],:p['n_area']]),np.abs(app_vecs_1[-1,:p['n_area'],:p['n_area']]),'o')
    ax[2].set_xlabel('eigenvectors')
    ax[2].set_ylabel('eigenvectors approximation')

def role_of_connection_spatial_localization(p_t,MACAQUE_CASE=1,SHUFFLE_TYPE=0,LINEAR_HIER=0,FULL_CASE=1,CONSENSUS_CASE=0):
    
    #if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
    #if SHUFFLE_TYPE==1:  #only permute the nonzero elements  
        
    p=p_t.copy()    
    dist_mat,full_dist_mat = generate_dist_matrix(p,MACAQUE_CASE=MACAQUE_CASE,CONSENSUS_CASE=CONSENSUS_CASE)
    
    p_ori,W_ori= genetate_net_connectivity(p,SHUFFLE_FLN=0,MACAQUE_CASE=MACAQUE_CASE,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVecs_ori, tau_ori = eig_decomposition(p_ori,W_ori,MACAQUE_CASE=MACAQUE_CASE)
    
    if FULL_CASE==1:
        lens=len(tau_ori)        
        theta_ori=np.zeros(lens)
        ipr_ori=np.zeros(lens)
        
        for i in range(lens):
            theta_ori[i]=THETA(eigVecs_ori[:,i],full_dist_mat)
            ipr_ori[i]=IPR(eigVecs_ori[:,i])  
            
        theta_ori=theta_ori[:p['n_area']]
        ipr_ori=ipr_ori[:p['n_area']]
        tau_ori=tau_ori[:p['n_area']]
        matcost_ori=np.sum(p['fln_mat']*full_dist_mat[:int(lens/2),:int(lens/2)])
        print('matcost_ori=',matcost_ori)
        
        n_trial=100
        
        record_theta_shuffled=np.zeros((p['n_area'],n_trial))
        record_ipr_shuffled=np.zeros((p['n_area'],n_trial))
        record_tau_shuffled=np.zeros((p['n_area'],n_trial))
        record_matcost_shuffled=np.zeros(n_trial)
        
        mean_theta=np.zeros(n_trial)
        mean_ipr=np.zeros(n_trial)
        range_tau=np.zeros(n_trial)
        
        fln_mat=p['fln_mat']
        for j in np.arange(n_trial):   
            theta_shuffle_temp=np.zeros(lens)
            ipr_shuffle_temp=np.zeros(lens)
        
            print('n_trial=',j)
            max_eigval=1
            while max_eigval>-1e-4:  #-5e-4:
                fln_shuffle_temp=matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
                max_eigval,W_shuffle_temp=unstability_detection(p,fln_shuffle_temp)
                print('max_eigval=',max_eigval)
            
            eigVecs_shuffle_temp, tau_shuffle_temp = eig_decomposition(p,W_shuffle_temp,MACAQUE_CASE=MACAQUE_CASE)
            plt.close('all')
            record_tau_shuffled[:,j]=tau_shuffle_temp[:p['n_area']]
            record_matcost_shuffled[j]=np.sum(fln_shuffle_temp*full_dist_mat[:int(lens/2),:int(lens/2)])
    
            for i in np.arange(int(lens/2)):
                theta_shuffle_temp[i]=THETA(eigVecs_shuffle_temp[:,i],full_dist_mat)
                ipr_shuffle_temp[i]=IPR(eigVecs_shuffle_temp[:,i])  
                
            record_theta_shuffled[:,j]=theta_shuffle_temp[:p['n_area']]
            record_ipr_shuffled[:,j]=ipr_shuffle_temp[:p['n_area']]
            
            mean_theta[j]=np.mean(record_theta_shuffled[:,j])
            mean_ipr[j]=np.mean(record_ipr_shuffled[:,j])
            range_tau[j]=np.std(record_tau_shuffled[:,j])
            #range_tau[j]=np.max(record_tau_shuffled[:,j])-np.min(record_tau_shuffled[:,j])    
    else:
        
        lens=int(len(tau_ori)/2)        
        theta_ori=np.zeros(lens)
        ipr_ori=np.zeros(lens)
        normalize_eigVecs_ori=normalize_matrix(eigVecs_ori[:lens,:lens],column=1)
                
        for i in range(lens):
            theta_ori[i]=THETA(normalize_eigVecs_ori[:,i],full_dist_mat[:lens,:lens])
            ipr_ori[i]=IPR(normalize_eigVecs_ori[:,i])  
            
        tau_ori=tau_ori[:p['n_area']]
        matcost_ori=np.sum(p['fln_mat']*full_dist_mat[:lens,:lens])
        print('matcost_ori=',matcost_ori)
        
        n_trial=100
        
        record_theta_shuffled=np.zeros((p['n_area'],n_trial))
        record_ipr_shuffled=np.zeros((p['n_area'],n_trial))
        record_tau_shuffled=np.zeros((p['n_area'],n_trial))
        record_matcost_shuffled=np.zeros(n_trial)
        
        mean_theta=np.zeros(n_trial)
        mean_ipr=np.zeros(n_trial)
        range_tau=np.zeros(n_trial)
        
        fln_mat=p['fln_mat']
        for j in np.arange(n_trial):   
            theta_shuffle_temp=np.zeros(lens)
            ipr_shuffle_temp=np.zeros(lens)
        
            print('n_trial=',j)
            max_eigval=1
            while max_eigval>-1e-4:  #-5e-4:
                fln_shuffle_temp=matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
                max_eigval,W_shuffle_temp=unstability_detection(p,fln_shuffle_temp)
                print('max_eigval=',max_eigval)
            
            eigVecs_shuffle_temp, tau_shuffle_temp = eig_decomposition(p,W_shuffle_temp,MACAQUE_CASE=MACAQUE_CASE)
            plt.close('all')
            record_tau_shuffled[:,j]=tau_shuffle_temp[:p['n_area']]
            record_matcost_shuffled[j]=np.sum(fln_shuffle_temp*full_dist_mat[:lens,:lens])
            
            normalize_eigVecs_shuffled=normalize_matrix(eigVecs_shuffle_temp[:lens,:lens],column=1)
            
            for i in np.arange(lens):
                theta_shuffle_temp[i]=THETA(normalize_eigVecs_shuffled[:,i],full_dist_mat[:lens,:lens])
                ipr_shuffle_temp[i]=IPR(normalize_eigVecs_shuffled[:,i])  
                
            record_theta_shuffled[:,j]=theta_shuffle_temp[:p['n_area']]
            record_ipr_shuffled[:,j]=ipr_shuffle_temp[:p['n_area']]
            
            mean_theta[j]=np.mean(record_theta_shuffled[:,j])
            mean_ipr[j]=np.mean(record_ipr_shuffled[:,j])
            #range_tau[j]=np.max(record_tau_shuffled[:,j])-np.min(record_tau_shuffled[:,j])    
            range_tau[j]=np.std(record_tau_shuffled[:,j])
            
    fig,ax=plt.subplots(2,3)
    ax[0,0].hist(theta_ori,15,facecolor='r',alpha=0.5)
    ax[0,0].set_xlabel('theta')
    ax[0,0].set_xlim([0,1])
    ax[0,0].set_title('mean=' + str(int(1e3*np.mean(theta_ori))/1e3) + '   std=' + str(int(1e3*np.std(theta_ori))/1e3))
    ax[0,1].hist(ipr_ori,15,facecolor='b',alpha=0.5)
    ax[0,1].set_xlabel('ipr')
    ax[0,1].set_xlim([0,1])
    ax[0,1].set_title('mean=' + str(int(1e3*np.mean(ipr_ori))/1e3) + '   std=' + str(int(1e3*np.std(ipr_ori))/1e3))
    ax[0,2].hist(tau_ori,15,facecolor='g',alpha=0.5)
    ax[0,2].set_xlabel('tau')
    ax[0,2].set_xlim(xmin=0)    
    ax[0,2].set_title('mean=' + str(int(1e3*np.mean(tau_ori))/1e3) + '   std=' + str(int(1e3*np.std(tau_ori))/1e3))
                    
    ax[1,0].hist(record_theta_shuffled.flatten(),15,facecolor='r',alpha=0.5)
    ax[1,0].set_xlabel('theta')
    ax[1,0].set_xlim([0,1])
    ax[1,0].set_title('mean=' + str(int(1e3*np.mean(record_theta_shuffled.flatten()))/1e3) + '   std=' + str(int(1e3*np.std(record_theta_shuffled.flatten()))/1e3))
    ax[1,1].hist(record_ipr_shuffled.flatten(),15,facecolor='b',alpha=0.5)
    ax[1,1].set_xlabel('ipr')
    ax[1,1].set_xlim([0,1])
    ax[1,1].set_title('mean=' + str(int(1e3*np.mean(record_ipr_shuffled.flatten()))/1e3) + '   std=' + str(int(1e3*np.std(record_ipr_shuffled.flatten()))/1e3))
    ax[1,2].hist(record_tau_shuffled.flatten(),15,facecolor='g',alpha=0.5)
    ax[1,2].set_xlabel('tau')
    ax[1,2].set_xlim(xmin=0)    
    ax[1,2].set_title('mean=' + str(int(1e3*np.mean(record_tau_shuffled.flatten()))/1e3) + '   std=' + str(int(1e3*np.std(record_tau_shuffled.flatten()))/1e3))
        
    fig,ax=plt.subplots(1,4,figsize=(40,10)) 
    ax[0].hist(mean_theta,15,rwidth=0.8,facecolor='r',alpha=0.5)
    ax[0].set_xlabel('mean theta')
    #ax[0].set_xlim([0,1])
    ax[0].axvline(np.mean(theta_ori))
    ax[1].hist(mean_ipr,15,rwidth=0.8,facecolor='b',alpha=0.5)
    ax[1].set_xlabel('mean ipr')
    #ax[1].set_xlim([0,1])
    ax[1].axvline(np.mean(ipr_ori))
    ax[2].hist(range_tau,15,rwidth=0.8,facecolor='g',alpha=0.5)
    ax[2].set_xlabel('tau standard deviation')
    #ax[2].set_xlim(xmin=0)    
    ax[2].axvline(np.std(tau_ori))
    ax[3].hist(record_matcost_shuffled,15,rwidth=0.8,facecolor='c',alpha=0.5)
    ax[3].set_xlabel('material cost')  
    ax[3].axvline(matcost_ori)
    
    fig,ax=plt.subplots(1,5,figsize=(50,10)) 
    ax[0].axvline(1)
    ax[0].hist(mean_theta/np.mean(theta_ori),15,rwidth=0.8,facecolor='r',alpha=0.5)
    ax[0].set_xlabel('normalized mean theta')
    #ax[0].set_xlim([0,1])
    ax[1].axvline(1)
    ax[1].hist(mean_ipr/np.mean(ipr_ori),15,rwidth=0.8,facecolor='b',alpha=0.5)
    ax[1].set_xlabel('normalized mean ipr')
    #ax[1].set_xlim([0,1])
    ax[2].axvline(1)
    ax[2].hist(range_tau/np.std(tau_ori),30,rwidth=0.8,facecolor='g',alpha=0.5)
    ax[2].set_xlabel('normalized std tau')
    #ax[2].set_xlim(xmin=0)    
    ax[3].axvline(1)
    ax[3].hist(mean_theta/np.mean(theta_ori)*mean_ipr/np.mean(ipr_ori),15,rwidth=0.8,facecolor='m',alpha=0.5)
    ax[3].set_xlabel('normalized theta*ipr')
    ax[4].axvline(1)
    ax[4].hist(mean_theta/np.mean(theta_ori)*mean_ipr/np.mean(ipr_ori)*range_tau/(np.max(tau_ori)-np.min(tau_ori)),30,rwidth=0.8,facecolor='c',alpha=0.5)
    ax[4].set_xlabel('normalized theta*ipr*range tau')

    
    shuffled_data={}
    shuffled_data['record_theta_shuffled']=record_theta_shuffled
    shuffled_data['record_ipr_shuffled']=record_ipr_shuffled
    shuffled_data['record_tau_shuffled']=record_tau_shuffled
    shuffled_data['mean_theta_shuffled']=mean_theta
    shuffled_data['mean_ipr_shuffled']=mean_ipr
    shuffled_data['range_tau_shuffled']=range_tau
    shuffled_data['theta_ori']=theta_ori
    shuffled_data['ipr_ori']=ipr_ori
    shuffled_data['tau_ori']=tau_ori
    
    filename = 'shuffled_data_'+str(MACAQUE_CASE)+'_'+str(SHUFFLE_TYPE)
    outfile = open(filename,'wb')
    pickle.dump(shuffled_data,outfile)
    outfile.close()
    
#shuffle FLN for multiple times and compute IPR and time constants    
#the time constant is defined as the decrease of response 5% above the baseline given a step current
def role_of_connection_by_shuffling_FLN(p_t,MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0):
    
    #if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
    #if SHUFFLE_TYPE==1:  #only permute the nonzero elements
    
    p=p_t.copy()
    fln_mat=p['fln_mat'].copy()
    
    tau_ori=run_stimulus_pulse_macaque(p,fln_mat)
    tau_loc=run_stimulus_pulse_macaque(p,np.zeros_like(fln_mat))
    #homogenerous fln_mat
    fln_mat_identical=fln_mat.copy()
    fln_mat_identical[fln_mat_identical>0]=np.mean(fln_mat_identical[fln_mat_identical>0])
    tau_identical=run_stimulus_pulse_macaque(p,fln_mat_identical)
    #network_graph_plot(p,tau_ori,MACAQUE_CASE=MACAQUE_CASE)
    # strong_link_statistics(p,tau_ori,tau_identical)
    ipr_eig_ori,ipr_green_ori=time_constant_localization_shuffle_fln(p,fln_mat,figname='time_matrix_original.png')
    
    mean_ipr_sigma_eig_ori=sigma_lambda_eigvector_localization_shuffle_fln(p,fln_mat,MACAQUE_CASE=MACAQUE_CASE)
    
    n_trial=100
    tau_shuffled=np.zeros((p['n_area'],n_trial))
    
    record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
    ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
    ipr_green_shuffled=np.zeros((p['n_area'],n_trial))
    KL_div_shuffled=np.zeros(n_trial)
    
    n_bin=11
    uniform_dist=np.ones(n_bin-1)/(n_bin-1)

    fig, ax = plt.subplots(1,2,figsize=(20,10))
    
    bins_tau_identical = np.linspace(np.min(tau_identical), np.max(tau_identical), n_bin)
    (counts_identical, bins, patch)=ax[0].hist(tau_identical, bins_tau_identical, alpha=0.5, label='tau homogeneous fln')
    ax[0].legend(loc='upper right')
        
    KL_div_identical=stats.entropy(counts_identical,uniform_dist,base=2)
    print('KL_div_identical=',KL_div_identical)
    
    bins_tau_ori = np.linspace(np.min(tau_ori), np.max(tau_ori), n_bin)
    (counts_ori, bins, patch)=ax[1].hist(tau_ori, bins_tau_ori, alpha=0.5, label='tau ori fln')
    ax[1].legend(loc='upper right')
    
    KL_div_ori=stats.entropy(counts_ori,uniform_dist,base=2)
    print('KL_div_ori=',KL_div_ori)
    
    mean_ipr_eig=np.zeros(n_trial)
    mean_ipr_green=np.zeros(n_trial)
    mean_ipr_sigma_eig=np.zeros(n_trial) 
    
    for j in np.arange(n_trial):   
        print('n_trial=',j)
        
        max_eigval=1
        while max_eigval>-1e-4:   #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
            fln_shuffled=matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
            max_eigval,_=unstability_detection(p,fln_shuffled)
            print('max_eigval=',max_eigval)
            
        tau_shuffled[:,j]=run_stimulus_pulse_macaque(p,fln_shuffled)
        record_fln_shuffled[:,:,j]=fln_shuffled
        p['fln_mat']=fln_shuffled
        # strong_link_statistics(p,tau_shuffled[:,j],tau_loc,figname='stats_'+str(j)+'.png')
        ipr_eig_shuffled[:,j],ipr_green_shuffled[:,j]=time_constant_localization_shuffle_fln(p,fln_shuffled,figname='time_matrix_'+str(j)+'.png')
        mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
        mean_ipr_green[j]=np.mean(ipr_green_shuffled[:,j])
        
        mean_ipr_sigma_eig[j]=sigma_lambda_eigvector_localization_shuffle_fln(p,fln_shuffled,MACAQUE_CASE=MACAQUE_CASE)
    #---------------------------------------------------------------------------------
    # compute KL divergence for shuffled tau
    #---------------------------------------------------------------------------------
        bins_tau_shuffled = np.linspace(np.min(tau_shuffled[:,j]), np.max(tau_shuffled[:,j]), n_bin)
        counts_shuffled, bins=np.histogram(tau_shuffled[:,j], bins_tau_shuffled)  #NOTE THE BIN WE CHOSE!
        KL_div_shuffled[j]=stats.entropy(counts_shuffled,uniform_dist,base=2)
        
    fig, ax = plt.subplots(figsize=(10,10))    
    ax.hist(KL_div_shuffled, alpha=0.3,color='b')   
    ax.axvline(KL_div_ori,color='r',label='with heterogeneous long range')    
    ax.axvline(KL_div_identical,color='k',label='with homogeneous long range')  
    ax.legend()
    ax.set_xlabel('KL divergence')
    ax.set_ylabel('counts')
    
    #---------------------------------------------------------------------------------
    # plot IPR for shuffled FLNs-I 
    #---------------------------------------------------------------------------------
    fig,ax=plt.subplots(1,4,figsize=(40,8)) 
    ax[0].hist(mean_ipr_eig,15,facecolor='r',alpha=0.5)
    ax[0].set_xlabel('mean ipr eig')
    #ax[0].set_xlim([0,1])
    ax[0].axvline(np.mean(ipr_eig_ori))
    ax[1].hist(mean_ipr_green,15,facecolor='b',alpha=0.5)
    ax[1].set_xlabel('mean ipr green')
    #ax[1].set_xlim([0,1])
    ax[1].axvline(np.mean(ipr_green_ori))
    
    ax[2].hist(mean_ipr_eig/np.mean(ipr_eig_ori),15,facecolor='r',alpha=0.5)
    ax[2].set_xlabel('normalized mean ipr eig')
    #ax[2].set_xlim([0,1])
    ax[2].axvline(1)
    ax[3].hist(mean_ipr_green/np.mean(ipr_green_ori),15,facecolor='b',alpha=0.5)
    ax[3].set_xlabel('normalized mean ipr green')
    #ax[3].set_xlim([0,1])
    ax[3].axvline(1)
    
    fig,ax=plt.subplots(1,2,figsize=(20,8)) 
    ax[0].hist(mean_ipr_sigma_eig,15,facecolor='r',alpha=0.5)
    ax[0].set_xlabel('mean ipr eig-SIGMA_CASE')
    #ax[0].set_xlim([0,1])
    ax[0].axvline(mean_ipr_sigma_eig_ori)
    
    ax[1].hist(mean_ipr_sigma_eig/mean_ipr_sigma_eig_ori,15,facecolor='r',alpha=0.5)
    ax[1].set_xlabel('normalized mean ipr eig-SIGMA_CASE')
    #ax[2].set_xlim([0,1])
    ax[1].axvline(1)
    
    #---------------------------------------------------------------------------------
    # plot IPR for shuffled FLNs-II
    #---------------------------------------------------------------------------------
    mean_ipr_eig=np.zeros(p['n_area']+1)
    mean_ipr_green=np.zeros(p['n_area']+1)

    top_90=np.zeros(p['n_area']+1)
    bottom_10=np.zeros(p['n_area']+1)
    top_95=np.zeros(p['n_area']+1)
    bottom_5=np.zeros(p['n_area']+1)
     
    plt.figure(figsize=(5,5))        
    ax = plt.axes()
    for k in np.arange(p['n_area']):
        mean_ipr_eig[k]=np.mean(ipr_eig_shuffled[k,:])
        sort_ipr_eig=np.sort(ipr_eig_shuffled[k,:])
        top_90[k]=sort_ipr_eig[int(0.9*n_trial)]
        bottom_10[k]=sort_ipr_eig[int(0.1*n_trial)]
        top_95[k]=sort_ipr_eig[int(0.95*n_trial)]
        bottom_5[k]=sort_ipr_eig[int(0.05*n_trial)]
        
        plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
        plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
    
    area_avg_ipr_eig=np.mean(ipr_eig_shuffled,0)
    mean_ipr_eig[p['n_area']]=np.mean(area_avg_ipr_eig)
    sort_area_avg_ipr_eig=np.sort(area_avg_ipr_eig)
    top_90[p['n_area']]=sort_area_avg_ipr_eig[int(0.9*n_trial)]
    bottom_10[p['n_area']]=sort_area_avg_ipr_eig[int(0.1*n_trial)]
    top_95[p['n_area']]=sort_area_avg_ipr_eig[int(0.95*n_trial)]
    bottom_5[p['n_area']]=sort_area_avg_ipr_eig[int(0.05*n_trial)]
    
    plt.vlines(p['n_area'], bottom_10[p['n_area']], top_90[p['n_area']],color="blue",alpha=0.3)
    plt.vlines(p['n_area'], bottom_5[p['n_area']], top_95[p['n_area']],color="blue",alpha=0.1)
        
    plt.plot(np.arange(p['n_area']+1),mean_ipr_eig,'.b',markersize=10)    
    plt.plot(np.arange(p['n_area']+1),top_90,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']+1),bottom_10,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']+1),top_95,'.b',markersize=6,alpha=0.1)  
    plt.plot(np.arange(p['n_area']+1),bottom_5,'.b',markersize=6,alpha=0.1)  
    ipr_eig_ori=np.append(ipr_eig_ori,np.mean(ipr_eig_ori))
    plt.plot(np.arange(p['n_area']+1),ipr_eig_ori,'.k',markersize=10)
    
    p['areas'].append('all')
    plt.xticks(np.arange(p['n_area']+1),p['areas'],rotation=90)
    #plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
    plt.ylabel('eigvector IPR')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.figure(figsize=(5,5))        
    ax = plt.axes()
    for k in np.arange(p['n_area']):
        mean_ipr_green[k]=np.mean(ipr_green_shuffled[k,:])
        sort_ipr_green=np.sort(ipr_green_shuffled[k,:])
        top_90[k]=sort_ipr_green[int(0.9*n_trial)]
        bottom_10[k]=sort_ipr_green[int(0.1*n_trial)]
        top_95[k]=sort_ipr_green[int(0.95*n_trial)]
        bottom_5[k]=sort_ipr_green[int(0.05*n_trial)]
        
        plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
        plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
        
        
    area_avg_ipr_green=np.mean(ipr_green_shuffled,0)
    mean_ipr_green[p['n_area']]=np.mean(area_avg_ipr_green)
    sort_area_avg_ipr_green=np.sort(area_avg_ipr_green)
    top_90[p['n_area']]=sort_area_avg_ipr_green[int(0.9*n_trial)]
    bottom_10[p['n_area']]=sort_area_avg_ipr_green[int(0.1*n_trial)]
    top_95[p['n_area']]=sort_area_avg_ipr_green[int(0.95*n_trial)]
    bottom_5[p['n_area']]=sort_area_avg_ipr_green[int(0.05*n_trial)]
    
    plt.vlines(p['n_area'], bottom_10[p['n_area']], top_90[p['n_area']],color="blue",alpha=0.3)
    plt.vlines(p['n_area'], bottom_5[p['n_area']], top_95[p['n_area']],color="blue",alpha=0.1)
        
    plt.plot(np.arange(p['n_area']+1),mean_ipr_green,'.b',markersize=10)    
    plt.plot(np.arange(p['n_area']+1),top_90,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']+1),bottom_10,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']+1),top_95,'.b',markersize=6,alpha=0.1)  
    plt.plot(np.arange(p['n_area']+1),bottom_5,'.b',markersize=6,alpha=0.1)  
    ipr_green_ori=np.append(ipr_green_ori,np.mean(ipr_green_ori))
    plt.plot(np.arange(p['n_area']+1),ipr_green_ori,'.k',markersize=10)
    
    plt.xticks(np.arange(p['n_area']+1),p['areas'],rotation=90)
    #plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
    plt.ylabel('Green func IPR')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    del p['areas'][-1]
    
    
    
    #---------------------------------------------------------------------------------
    # plot time constants for shuffled FLNs
    #---------------------------------------------------------------------------------
    mean_tau=np.zeros(p['n_area'])
    top_90=np.zeros(p['n_area'])
    bottom_10=np.zeros(p['n_area'])
    top_95=np.zeros(p['n_area'])
    bottom_5=np.zeros(p['n_area'])
     
    plt.figure(figsize=(5,5))        
    ax = plt.axes()
    for k in np.arange(p['n_area']):
        mean_tau[k]=np.mean(tau_shuffled[k,:])
        sort_tau=np.sort(tau_shuffled[k,:])
        top_90[k]=sort_tau[int(0.9*n_trial)]
        bottom_10[k]=sort_tau[int(0.1*n_trial)]
        top_95[k]=sort_tau[int(0.95*n_trial)]
        bottom_5[k]=sort_tau[int(0.05*n_trial)]
        
        plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
        plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
        
        
    plt.plot(np.arange(p['n_area']),mean_tau,'.b',markersize=10)    
    plt.plot(np.arange(p['n_area']),top_90,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']),bottom_10,'.b',markersize=8,alpha=0.3)  
    plt.plot(np.arange(p['n_area']),top_95,'.b',markersize=6,alpha=0.1)  
    plt.plot(np.arange(p['n_area']),bottom_5,'.b',markersize=6,alpha=0.1)  
    plt.plot(np.arange(p['n_area']),tau_ori,'.k',markersize=10)
    
    plt.xticks(np.arange(p['n_area']),p['areas'],rotation=90)
    plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
    plt.ylabel('$T_{delay}$ (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return record_fln_shuffled, tau_shuffled


#visualize the graph of areas and connections for macaque network
def network_graph_plot(p_t,par=None,MACAQUE_CASE=1):

    if MACAQUE_CASE==0:
        return
    
    colormode=0
        
    p=p_t.copy()
    
    if par is None:
        par=np.ones(p['n_area'],dtype=int)
        
    #position of brain areas
    a = [1.6, 2.3, 3.2, 4.1, 4.7, 12.8, 6.9, 12.2, 3.6, 7.9, 9.0, 5.0, 5.0, 16.1, 18.2, 13.2, 14.7, 10.7, 6.1, 7.3, 6.8, 6.1, 13.0, 6.9, 11.3, 15.6, 16.1, 9.2, 15.0]
    b = [8.1, 10.0, 8.1, 13.5, 9.5, 12.5, 13.8, 12.0, 6.1, 10.0, 14.1, 9.7, 14.0, 12.0, 11.7, 10.9, 11.8, 9.7, 3.6, 6.9, 15.0, 11.3, 15.5, 6.3, 7.5, 14.0, 13.7, 4.9, 12.5]
    
    plt.figure(figsize=(15,10))
    plt.scatter(a,b, s=abs(par)*500,color='red')
    
    for i in range(29):
        if colormode==0:
            plt.text(a[i],b[i],p['areas'][i]+'-'+str(int(par[i]*1e2)/1e2), family='serif', style='italic', ha='right', wrap=True)
        else:
            plt.text(a[i],b[i],p['areas'][i]+'-'+str(round(p['hier_vals'][i],3))+'-'+str(round(par[i],5)), family='serif', style='italic', ha='right', wrap=True)

    fln_flat=p['fln_mat'].flatten()
    fln_flat=fln_flat[fln_flat>0]
    idx_fln=np.argsort(fln_flat)
    fln_sort=np.sort(fln_flat)
    thr_fln=fln_sort[int(len(idx_fln)*0.67)]
    print('thr_fln=',thr_fln)
    print('num_strong_link=',len(fln_flat[fln_flat>thr_fln]))

    #feedforward
    w = np.empty([p['n_area'],p['n_area']])
    paramuEEsp = 0.09
    for i in range(p['n_area']):
        for j in range(i+1,p['n_area']):
            w[j,i] = (1 + p['eta'] * p['hier_vals'][j]) * paramuEEsp * p['fln_mat'][j,i]
            x = [[a[i],a[j]]]
            y = [[b[i],b[j]]]
            if p['fln_mat'][j,i] > thr_fln:
                    if colormode==0:
                        plt.plot(x[0],y[0],color='k',linewidth = 50*(w[j,i]+0.015),alpha=0.3)
                    else:
                        plt.plot(x[0],y[0],color='r',linewidth = 50*(w[j,i]+0.015),alpha=0.5)
                    
    
    #feedback
    for i in range(p['n_area']):
        for j in range(i+1,p['n_area']):
            w[i,j] = (1 + p['eta'] * p['hier_vals'][i]) * paramuEEsp * p['fln_mat'][i,j]
            x = [[a[i],a[j]]]
            y = [[b[i],b[j]]]
            if p['fln_mat'][i,j] > thr_fln:
                if colormode==0:
                    plt.plot(x[0],y[0],'k',linewidth = 50*(w[i,j]+0.015),alpha=0.3)
                else:
                    if p['fln_mat'][j,i] > thr_fln:
                        plt.plot(x[0],y[0],'--b',linewidth = 50*(w[i,j]+0.015),alpha=0.5)
                    else:
                        plt.plot(x[0],y[0],color='b',linewidth = 50*(w[i,j]+0.015),alpha=0.5)
      
    
#plot statistics regarding the strong fln only   
def strong_link_statistics(p_t,tau,tau_loc,figname='stats_original.png'):
    
    p=p_t.copy()
    
    fln_flat=p['fln_mat'].flatten()
    fln_flat=fln_flat[fln_flat>0]
    idx_fln=np.argsort(fln_flat)
    fln_sort=np.sort(fln_flat)
    thr_fln=fln_sort[int(len(idx_fln)*0.67)]
    print('thr_fln=',thr_fln)
    print('num_strong_link=',len(fln_flat[fln_flat>thr_fln]))
    
    forward_count=0
    backward_count=0 
    bidirect_count=0
    
    #feedforward
    for i in range(p['n_area']):
        for j in range(i+1,p['n_area']):
            if p['fln_mat'][j,i] > thr_fln:
                forward_count=forward_count+1
                
    #feedback
    for i in range(p['n_area']):
        for j in range(i+1,p['n_area']):
            if p['fln_mat'][i,j] > thr_fln:
                backward_count=backward_count+1
                if p['fln_mat'][j,i] > thr_fln:
                    bidirect_count=bidirect_count+1
    
    print('forward_count=',forward_count)    
    print('backward_count=',backward_count)    
    print('bidirect_count=',bidirect_count)    
    
    y_fln=np.zeros(p['n_area']**2-p['n_area'])
    y_dtau=np.zeros_like(y_fln)
    y_dtau_loc=np.zeros_like(y_fln)
    y_dh=np.zeros_like(y_fln)
    
    fig, ax = plt.subplots(3,2,figsize=(20,20))
    k=0
    for i in np.arange(p['n_area']):
        for j in np.arange(p['n_area']):
            if i!=j:
                y_fln[k]=p['fln_mat'][j,i]
                y_dtau[k]=tau[i]-tau[j]
                y_dtau_loc[k]=tau_loc[i]-tau_loc[j]
                y_dh[k]=p['hier_vals'][i]-p['hier_vals'][j]
                k=k+1

    y_fln[y_fln==0]=1e-6
    
    
    #---------------------------------------------------------------------------------
    # plot result
    #--------------------------------------------------------------------------------- 
    ax[0,0].scatter(y_fln,y_dh)
    ax[0,0].scatter(y_fln[y_fln>thr_fln],y_dh[y_fln>thr_fln],c='r')
    ax[0,0].set_title('fln vs $\Delta$h')
    ax[0,0].set_xlabel('fln')
    ax[0,0].set_ylabel(r'$\Delta$h')
    ax[0,0].set_xscale('log')
    ax[0,0].set_xlim((1e-6,1e1))
    
    ax[0,1].scatter(y_fln,y_dtau)
    ax[0,1].scatter(y_fln[y_fln>thr_fln],y_dtau[y_fln>thr_fln],c='r')
    ax[0,1].set_title('fln vs $\Delta$tau')
    ax[0,1].set_xlabel('fln')
    ax[0,1].set_ylabel(r'$\Delta$tau')
    ax[0,1].set_xscale('log')
    ax[0,1].set_xlim((1e-6,1e1))
    
    ax[1,0].scatter(tau_loc,tau)
    ax[1,0].set_title('tau_loc vs tau')
    ax[1,0].set_xlabel('tau_loc')
    ax[1,0].set_ylabel('tau')
    
    ax[1,1].scatter(y_dtau_loc,y_dtau)
    ax[1,1].set_title(r'$\Delta$tau_loc vs $\Delta$tau')
    ax[1,1].set_xlabel(r'$\Delta$tau_loc')
    ax[1,1].set_ylabel(r'$\Delta$tau')
    
    bins_tau = np.linspace(np.min(tau), np.max(tau), 10)
    bins_tau_loc = np.linspace(np.min(tau_loc), np.max(tau_loc), 10)
    ax[2,0].hist(tau, bins_tau, alpha=0.5, label='tau')
    ax[2,0].hist(tau_loc, bins_tau_loc, alpha=0.5, label='tau_loc')
    ax[2,0].legend(loc='upper right')
    
    bins_tau = np.linspace(np.min(y_dtau), np.max(y_dtau), 10)
    bins_tau_loc = np.linspace(np.min(y_dtau_loc), np.max(y_dtau_loc), 10)
    ax[2,1].hist(y_dtau, bins_tau, alpha=0.5, label='tau')
    ax[2,1].hist(y_dtau_loc, bins_tau_loc, alpha=0.5, label='tau_loc')
    ax[2,1].legend(loc='upper right')
    
    fig.savefig('result/'+figname)   
    plt.close(fig)
    


def ipr_from_regular_to_original_networks(p_t,full_dist_mat_t,MACAQUE_CASE=1,SHUFFLE_TYPE=5,FULL_CASE=1,CONSENSUS_CASE=0):
    
    if FULL_CASE!=1 or (SHUFFLE_TYPE<5 and SHUFFLE_TYPE!=6):
        raise SystemExit('FULL_CASE has to be 1 and SHUFFLE_TYPE has to be 5 or 6 for now!!')
        
    p=p_t.copy()
    full_dist_mat=full_dist_mat_t.copy()
        
    p_ori,_ = genetate_net_connectivity(p,SHUFFLE_FLN=0,MACAQUE_CASE=MACAQUE_CASE,CONSENSUS_CASE=CONSENSUS_CASE)
    p_reg,_ = genetate_net_connectivity(p,SHUFFLE_FLN=1, SHUFFLE_TYPE=SHUFFLE_TYPE, MACAQUE_CASE=MACAQUE_CASE,CONSENSUS_CASE=CONSENSUS_CASE)
    
    if SHUFFLE_TYPE==5:            
        partial_dist=full_dist_mat[:p_ori['n_area'],:p_ori['n_area']]
    if SHUFFLE_TYPE==6:
        hier_mat=np.zeros(p_ori['n_area'],p_ori['n_area'])
        for i in np.arange():
            for j in np.arange():
                hier_mat[i,j]=np.abs(p_ori['hier_vals'][i]-p_ori['hier_vals'][j])
                
    n_link=p_ori['n_area']**2-p_ori['n_area']
    n_step=10
    step_size=int(n_link/n_step)
    lens=len(np.arange(0,n_link,step_size))
    mean_theta=np.zeros(lens)
    mean_ipr=np.zeros(lens)
    tau_range=np.zeros(lens)
    count=0
    
    mask=~np.eye(p['n_area'],dtype=bool)
                  
    for i in np.arange(0,n_link,step_size):
        fln_shuffle=partial_swap(p_reg['fln_mat'],p_ori['fln_mat'],mask,(0,i))
        fig,ax=plt.subplots()
        if SHUFFLE_TYPE==5:
            ax.scatter(partial_dist.flatten(),fln_shuffle)
        if SHUFFLE_TYPE==6:
            ax.scatter(hier_mat.flatten(),fln_shuffle)
        ax.set_title(['step='+str(i)])
        max_eigval,W_shuffle=unstability_detection(p_reg,fln_shuffle)
        print('max_eigval=',max_eigval)        
        eigVecs_shuffle, tau_shuffle = eig_decomposition(p_ori,W_shuffle,MACAQUE_CASE=MACAQUE_CASE,CLOSE_FIG=1)
        ipr,_=IPR_MATRIX(eigVecs_shuffle)
        theta,_=THETA_MATRIX(eigVecs_shuffle,full_dist_mat)
        mean_theta[count]=np.mean(theta[:p_ori['n_area']])
        mean_ipr[count]=np.mean(ipr[:p_ori['n_area']])
        tau_range[count]=tau_shuffle[0]-tau_shuffle[p_ori['n_area']]
        count=count+1

    fig,ax=plt.subplots(1,3,figsize=(30,10))
    ax[0].plot(np.arange(0,n_link,step_size)/n_link,mean_ipr)
    ax[0].set_ylabel('mean ipr')
    ax[0].set_xlabel('swap percentage')
    
    ax[1].plot(np.arange(0,n_link,step_size)/n_link,mean_theta)
    ax[1].set_ylabel('mean theta')
    ax[1].set_xlabel('swap percentage')
    
    ax[2].plot(np.arange(0,n_link,step_size)/n_link,tau_range)
    ax[2].set_ylabel('tau range')
    ax[2].set_xlabel('swap percentage')
        
        
def partial_swap(A,B,mask=None, order_range=None):
    # create buffer for masked elements, 1d array here
    A_masked = A[mask]
    B_masked = B[mask]
    # sort 1d masked elements
    A_index = np.flip(np.argsort(A_masked))
    B_index = np.flip(np.argsort(B_masked))
    # swap A masked
    if order_range is None:
        A_masked[A_index], A_masked[B_index] = A_masked[B_index], A_masked[A_index]
    else:
        A_masked[A_index[order_range[0]:order_range[1]]], A_masked[B_index[order_range[0]:order_range[1]]] = A_masked[B_index[order_range[0]:order_range[1]]], A_masked[A_index[order_range[0]:order_range[1]]]

    A_new = A.copy()
    # update original matrix
    A_new[mask] = A_masked
    return A_new    

#------------------------------------------------------------------------------
#functions not often used
#------------------------------------------------------------------------------
def theta_ipr_plot_original_shuffled_comparision(p_t,full_dist_mat_t,MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0,FULL_CASE=1,CONSENSUS_CASE=0):

    p=p_t.copy()
    full_dist_mat=full_dist_mat_t.copy()
    p1,W1 = genetate_net_connectivity(p,LINEAR_HIER=LINEAR_HIER,IDENTICAL_HIER=0,ZERO_FLN=0,IDENTICAL_FLN=0,SHUFFLE_FLN=0, STRONG_GBA=0,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=CONSENSUS_CASE)
    p2,W2 = genetate_net_connectivity(p,LINEAR_HIER=LINEAR_HIER,IDENTICAL_HIER=0,ZERO_FLN=0,IDENTICAL_FLN=0,SHUFFLE_FLN=1, SHUFFLE_TYPE=SHUFFLE_TYPE, STRONG_GBA=0,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVecs_reorder1, tau_reorder1 = eig_decomposition(p1,W1,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0)
    eigVecs_reorder2, tau_reorder2 = eig_decomposition(p2,W2,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0)
    
    Sigma1,Lambda1=get_Sigma_Lambda_matrix(p1,W1,MACAQUE_CASE=MACAQUE_CASE)
    Sigma2,Lambda2=get_Sigma_Lambda_matrix(p2,W2,MACAQUE_CASE=MACAQUE_CASE)
    
    #-----------------plot the effective perturbation elements in SIGMA LAMBDA cases-----------------
    F_EE1=Sigma1
    F_EE2=Sigma2
    
    min_r=1e-4
    max_r1=np.max(np.abs(Sigma1+Lambda1))
    max_r2=np.max(np.abs(Sigma2+Lambda2))
    
    Sigma_Lambda1=np.abs(Sigma1+Lambda1)
    Sigma_Lambda2=np.abs(Sigma2+Lambda2)
    # Sigma_Lambda1[Sigma_Lambda1<min_r]=0
    # Sigma_Lambda2[Sigma_Lambda2<min_r]=0
    
    fig,ax=plt.subplots(1,2,figsize=(30,10))
    f=ax[0].pcolormesh(Sigma_Lambda1,cmap='hot',norm=LogNorm(vmin=min_r, vmax=max_r1))
    fig.colorbar(f,ax=ax[0],pad=0.15)
    ax[0].set_title('SIGMA1+LAMBDA1')
    ax[0].invert_yaxis()
    f=ax[1].pcolormesh(Sigma_Lambda2,cmap='hot',norm=LogNorm(vmin=min_r, vmax=max_r2))
    fig.colorbar(f,ax=ax[1],pad=0.15)
    ax[1].set_title('SIGMA2+LAMBDA2')
    ax[1].invert_yaxis()
    
    eigvals_norder1,eigVecs_norder_sigma1 = np.linalg.eig(Sigma1+Lambda1)
    eigvals_norder2,eigVecs_norder_sigma2 = np.linalg.eig(Sigma2+Lambda2)
    
    le=len(p1['hier_vals'])
    id_mat=np.eye(le)
    ind1=np.argsort(-np.real(eigvals_norder1))
    per_mat1=id_mat.copy()
    ind2=np.argsort(-np.real(eigvals_norder2))
    per_mat2=id_mat.copy()
    for i in np.arange(le):
        per_mat1[i,:]=id_mat.copy()[ind1[i],:]
        per_mat2[i,:]=id_mat.copy()[ind2[i],:]
    
    eigvecs_order_sigma1=eigVecs_norder_sigma1@(np.linalg.inv(per_mat1))  
    eigvecs_order_sigma2=eigVecs_norder_sigma2@(np.linalg.inv(per_mat2))  
    
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(30,10))
    f=ax1.pcolormesh(np.abs(eigvecs_order_sigma1),cmap='hot',norm=LogNorm(vmin=1e-3, vmax=1))
    fig.colorbar(f,ax=ax1,pad=0.15)
    ax1.set_title('eigen SIGMA1')
    ax1.invert_yaxis()
    ax1.invert_xaxis()

    f=ax2.pcolormesh(np.abs(eigvecs_order_sigma2),cmap='hot',norm=LogNorm(vmin=1e-3, vmax=1))
    fig.colorbar(f,ax=ax2,pad=0.15)
    ax2.set_title('eigen SIGMA2')
    ax2.invert_yaxis()
    ax2.invert_xaxis()    
    
    _,mean_ipr1=IPR_MATRIX(eigvecs_order_sigma1)
    _,mean_ipr2=IPR_MATRIX(eigvecs_order_sigma2)
    print('mean_ipr1=',mean_ipr1)
    print('mean_ipr2=',mean_ipr2)
    print('ipr_ratio_sigma_case=',mean_ipr2/mean_ipr1)
    
    
    fig,ax=plt.subplots(1,3,figsize=(30,10))
    sum_fln1=np.sum(Sigma1,axis=1)
    sum_fln2=np.sum(Sigma2,axis=1)
    ax[0].scatter(np.diag(np.abs(Lambda1)),sum_fln1,30,label='original',alpha=0.5)
    ax[0].scatter(np.diag(np.abs(Lambda2)),sum_fln2,30,label='shuffled',alpha=0.5)
    ax[0].set_xlabel('Lambda')
    ax[0].set_ylabel('summed Sigma for each node-SIGMA LAMBDA')
    ax[0].legend()
    
    hier_diff_mat=np.zeros_like(F_EE1)     #F_EE_ij is proportional to FLN_ij(1+eta h_i)
    per_mat1=np.zeros_like(F_EE1) 
    per_mat2=np.zeros_like(F_EE2) 
    le=len(p1['hier_vals'])
    for i in np.arange(le):
        for j in np.arange(le):
            factor=np.abs(Lambda1[i,i]-Lambda1[j,j])
            if factor>0:
                hier_diff_mat[i,j]=factor
                per_mat1[i,j]=F_EE1[i,j]/factor
                per_mat2[i,j]=F_EE2[i,j]/factor
            
    ax[1].scatter(hier_diff_mat[F_EE1!=0],F_EE1[F_EE1!=0],30,label='original',alpha=0.5)
    ax[1].scatter(hier_diff_mat[F_EE2!=0],F_EE2[F_EE2!=0],30,label='shuffled',alpha=0.5)
    ax[1].set_xlabel('Lambda difference-SIGMA LAMBDA')
    ax[1].set_ylabel('Sigma / Diff Lambda')
    ax[1].set_ylim(ymin=0)  
    ax[1].legend()
    
    mean_FEE1=np.mean(F_EE1[F_EE1!=0])
    mean_FEE2=np.mean(F_EE2[F_EE2!=0])
    print('mean_FEE1=',mean_FEE1)
    print('mean_FEE2=',mean_FEE2)
    
    ax[2].hist(per_mat1[per_mat1!=0],bins=100,label='original',alpha=0.5)
    ax[2].hist(per_mat2[per_mat2!=0],bins=100,label='shuffled',alpha=0.5)
    ax[2].set_xlabel('effective perturabation elements-SIGMA LAMBDA')
    ax[2].set_yscale('log')
    ax[2].legend()
    
    mean_per_mat1=np.mean(per_mat1[per_mat1!=0])
    mean_per_mat2=np.mean(per_mat2[per_mat2!=0])
    print('mean_per_mat1=',mean_per_mat1)
    print('mean_per_mat2=',mean_per_mat2)
    #----------end of plot the effective perturbation elements in SIGMA LAMBDA cases-----------------
    
    #-----------------plot the effective perturbation elements-----------------
    F_EE1=W1.copy()[0::2,0::2]
    np.fill_diagonal(F_EE1,0)  
    F_EE2=W2.copy()[0::2,0::2]
    np.fill_diagonal(F_EE2,0)  
    
    fig,ax=plt.subplots(1,3,figsize=(30,10))
    sum_fln1=np.sum(p1['fln_mat'],axis=1)
    sum_fln2=np.sum(p2['fln_mat'],axis=1)
    ax[0].scatter(p1['hier_vals'],sum_fln1,30,label='original',alpha=0.5)
    ax[0].scatter(p2['hier_vals'],sum_fln2,30,label='shuffled',alpha=0.5)
    ax[0].set_xlabel('hierarchy')
    ax[0].set_ylabel('summed FLN for each node')
    ax[0].legend()
    
    hier_diff_mat=np.zeros_like(F_EE1)     #F_EE_ij is proportional to FLN_ij(1+eta h_i)
    per_mat1=np.zeros_like(F_EE1) 
    per_mat2=np.zeros_like(F_EE2) 
    le=len(p1['hier_vals'])
    for i in np.arange(le):
        for j in np.arange(le):
            factor=np.abs(p1['hier_vals'][i]-p1['hier_vals'][j])
            if factor>0:
                hier_diff_mat[i,j]=factor
                per_mat1[i,j]=F_EE1[i,j]/factor
                per_mat2[i,j]=F_EE2[i,j]/factor
            
    ax[1].scatter(hier_diff_mat[F_EE1>0],F_EE1[F_EE1>0],30,label='original',alpha=0.5)
    ax[1].scatter(hier_diff_mat[F_EE2>0],F_EE2[F_EE2>0],30,label='shuffled',alpha=0.5)
    ax[1].set_xlabel('hierarchy difference')
    ax[1].set_ylabel('F_EE')        
    ax[1].legend()
    
    ax[2].hist(per_mat1[per_mat1>0],bins=100,label='original',alpha=0.5)
    ax[2].hist(per_mat2[per_mat2>0],bins=100,label='shuffled',alpha=0.5)
    ax[2].set_xlabel('effective perturabation elements')
    ax[2].set_yscale('log')
    ax[2].legend()
    #----------end of plot the effective perturbation elements-----------------
    
    if FULL_CASE==1:
        theta1,_=THETA_MATRIX(eigVecs_reorder1,full_dist_mat)
        ipr1,_=IPR_MATRIX(eigVecs_reorder1)
        
        theta2,_=THETA_MATRIX(eigVecs_reorder2,full_dist_mat)
        ipr2,_=IPR_MATRIX(eigVecs_reorder2)
        
        lens=int(len(tau_reorder1))
        x1=np.flip(ipr1[:int(lens/2)])
        y1=np.flip(theta1[:int(lens/2)])
        x2=np.flip(ipr2[:int(lens/2)])
        y2=np.flip(theta2[:int(lens/2)])
        
    else:
        
        lens=int(len(tau_reorder1)/2)        
        theta1=np.zeros(lens)
        ipr1=np.zeros(lens)
        
        normalize_eigVecs_reorder1=normalize_matrix(eigVecs_reorder1[:lens,:lens],column=1)
        normalize_eigVecs_reorder2=normalize_matrix(eigVecs_reorder2[:lens,:lens],column=1)
        
        #-----------------------plot normalized submatrix----------------------
   
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(30,10))
        f=ax1.pcolormesh(normalize_eigVecs_reorder1,cmap='hot')
        fig.colorbar(f,ax=ax1,pad=0.15)
        ax1.set_title('normalize_eigVecs_reorder1')
        
        ax1.invert_yaxis()
        ax1.invert_xaxis()
        
        x = np.arange(lens) # xticks
        y = np.arange(p['n_area']) # yticks
        xlim = (0,lens)
        ylim = (0,p['n_area'])
        
        yticklabels_odd=p['areas'][1::2]
        yticklabels_even=p['areas'][::2]
        
        # set original ticks and ticklabels
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        ax1.set_xticks(x[::5])
        ax1.invert_xaxis()
           
        ax1.set_xticklabels(tau_reorder1[:lens:5])
        ax1.set_yticks(y[::2])
        ax1.set_yticklabels(yticklabels_even)
        ax1.invert_yaxis()
        # rotate xticklabels to 90 degree
        plt.setp(ax1.get_xticklabels(), rotation=90)
        
        # second y axis
        ax3 = ax1.twinx()
        ax3.set_ylim(ylim)
        ax3.set_yticks(y[1::2])
        ax3.set_yticklabels(yticklabels_odd)
        ax3.invert_yaxis()   
        
        
        f=ax2.pcolormesh(normalize_eigVecs_reorder2,cmap='hot')
        fig.colorbar(f,ax=ax2,pad=0.15)
        ax2.set_title('normalize_eigVecs_reorder2')
        
        ax2.invert_yaxis()
        ax2.invert_xaxis()
        
        x = np.arange(lens) # xticks
        y = np.arange(p['n_area']) # yticks
        xlim = (0,lens)
        ylim = (0,p['n_area'])
        
        yticklabels_odd=p['areas'][1::2]
        yticklabels_even=p['areas'][::2]
        
        # set original ticks and ticklabels
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        ax2.set_xticks(x[::5])
        ax2.invert_xaxis()
           
        ax2.set_xticklabels(tau_reorder1[:lens:5])
        ax2.set_yticks(y[::2])
        ax2.set_yticklabels(yticklabels_even)
        ax2.invert_yaxis()
        # rotate xticklabels to 90 degree
        plt.setp(ax2.get_xticklabels(), rotation=90)
        
        # second y axis
        ax4 = ax1.twinx()
        ax4.set_ylim(ylim)
        ax4.set_yticks(y[1::2])
        ax4.set_yticklabels(yticklabels_odd)
        ax4.invert_yaxis()   
        
        #----------------end of plot normalized submatrix----------------------
       
        theta1,_=THETA_MATRIX(normalize_eigVecs_reorder1,full_dist_mat[:lens,:lens])
        ipr1,_=IPR_MATRIX(normalize_eigVecs_reorder1)
    
        theta2,_=THETA_MATRIX(normalize_eigVecs_reorder2,full_dist_mat[:lens,:lens])
        ipr2,_=IPR_MATRIX(normalize_eigVecs_reorder2)    
        
        x1=np.flip(ipr1)
        y1=np.flip(theta1)
        x2=np.flip(ipr2)
        y2=np.flip(theta2)   
    
    fig,axScatter=plt.subplots(figsize=(10,10))
    axScatter.scatter(x1,y1,100,color='r',alpha=0.5,label='original network')
    axScatter.scatter(x2,y2,100,color='b',alpha=0.5,label='shuffled network')
    plt.legend(loc='lower right')
    
    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 2.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 2.2, pad=0.1, sharey=axScatter)
    
    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    
    n_bin=15
    x_start=min(np.append(x1,x2))-0.05
    y_start=min(np.append(y1,y2))-0.05
    x_end=max(np.append(x1,x2))+0.05
    y_end=max(np.append(y1,y2))+0.05
    x_width=(x_end-x_start)/n_bin
    y_width=(y_end-y_start)/n_bin

    bins_x = np.arange(x_start, x_end, x_width)
    bins_y = np.arange(y_start, y_end, y_width)
    axHistx.hist(x1, bins=bins_x,rwidth=0.8,color='r',alpha=0.5)
    axHisty.hist(y1, bins=bins_y,rwidth=0.8,color='r',alpha=0.5,orientation='horizontal')
        
    bins_x = np.arange(x_start, x_end, x_width)
    bins_y = np.arange(y_start, y_end, y_width)
    axHistx.hist(x2, bins=bins_x,rwidth=0.8,color='b',alpha=0.5)
    axHisty.hist(y2, bins=bins_y,rwidth=0.8,color='b',alpha=0.5, orientation='horizontal')
    
    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    axScatter.set_ylim([0,1.05])
    axScatter.set_xlim([0,1.05])
    axScatter.set_xlabel('IPR')
    axScatter.set_ylabel(r'$\theta$')
    
    localization_ind=local_index(x1,y1,x2,y2)
    print('ipr loc=',np.mean(x2)/np.mean(x1))
    print('theta loc=',np.mean(y2)/np.mean(y1))
    print('localization_ind=',localization_ind)
    
    
    #------------------------compute SIGMA+LAMBDA eigenvector-------------------------
    

def theta_ipr_plot_weak_strong_GBA_comparision(p_t,full_dist_mat_t,MACAQUE_CASE=1,CONSENSUS_CASE=0):

    p=p_t.copy()
    full_dist_mat=full_dist_mat_t.copy()
    p1,W1 = genetate_net_connectivity(p,LINEAR_HIER=0,IDENTICAL_HIER=0,ZERO_FLN=0,IDENTICAL_FLN=0,SHUFFLE_FLN=0, STRONG_GBA=0,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=CONSENSUS_CASE)
    p2,W2 = genetate_net_connectivity(p,LINEAR_HIER=0,IDENTICAL_HIER=0,ZERO_FLN=0,IDENTICAL_FLN=0,SHUFFLE_FLN=0, STRONG_GBA=1,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVecs_reorder1, tau_reorder1 = eig_decomposition(p1,W1,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0)
    eigVecs_reorder2, tau_reorder2 = eig_decomposition(p2,W2,MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0)
        
    
    lens=len(tau_reorder1)        
    theta1,_=THETA_MATRIX(eigVecs_reorder1,full_dist_mat)
    ipr1,_=IPR_MATRIX(eigVecs_reorder1)
    
    theta2,_=THETA_MATRIX(eigVecs_reorder2,full_dist_mat)
    ipr2,_=IPR_MATRIX(eigVecs_reorder2)
    
    x1=np.flip(ipr1[:int(lens/2)])
    y1=np.flip(theta1[:int(lens/2)])
    x2=np.flip(ipr2[:int(lens/2)])
    y2=np.flip(theta2[:int(lens/2)])
    
    fig,axScatter=plt.subplots(figsize=(10,10))
    axScatter.scatter(x1,y1,100,color='r',alpha=0.5,label='weak GBA')
    axScatter.scatter(x2,y2,100,color='b',alpha=0.5,label='strong GBA')
    plt.legend()
    
    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 2.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 2.2, pad=0.1, sharey=axScatter)
    
    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    
    n_bin=15
    x_start=min(np.append(x1,x2))-0.05
    y_start=min(np.append(y1,y2))-0.05
    x_end=max(np.append(x1,x2))+0.05
    y_end=max(np.append(y1,y2))+0.05
    x_width=(x_end-x_start)/n_bin
    y_width=(y_end-y_start)/n_bin

    bins_x = np.arange(x_start, x_end, x_width)
    bins_y = np.arange(y_start, y_end, y_width)
    axHistx.hist(x1, bins=bins_x,rwidth=0.8,color='r',alpha=0.5)
    axHisty.hist(y1, bins=bins_y,rwidth=0.8,color='r',alpha=0.5,orientation='horizontal')
        
    bins_x = np.arange(x_start, x_end, x_width)
    bins_y = np.arange(y_start, y_end, y_width)
    axHistx.hist(x2, bins=bins_x,rwidth=0.8,color='b',alpha=0.5)
    axHisty.hist(y2, bins=bins_y,rwidth=0.8,color='b',alpha=0.5, orientation='horizontal')
    
    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    axScatter.set_ylim([0,1.05])
    axScatter.set_xlim([0,0.72])
    axScatter.set_xlabel('IPR')
    axScatter.set_ylabel(r'$\theta$')
    
    localization_ind=local_index(x1,y1,x2,y2)
    print('localization_ind=',localization_ind)

# plot eigenvalue of different connectivity matrices     
def plot_eigenvalue_under_various_conditions(p_t,CONSENSUS_CASE=0):
    p=p_t.copy()
    _,W1=genetate_net_connectivity(p,IDENTICAL_HIER=0,ZERO_FLN=0,MACAQUE_CASE=0,STRONG_GBA=1,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVals_1,_ = np.linalg.eig(W1)
    eigVals_1=np.sort(np.real(eigVals_1))
    tau_1=-1/eigVals_1
    
    _,W2=genetate_net_connectivity(p,IDENTICAL_HIER=0,ZERO_FLN=1,MACAQUE_CASE=0,STRONG_GBA=1,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVals_2,_ = np.linalg.eig(W2)
    eigVals_2=np.sort(np.real(eigVals_2))
    tau_2=-1/eigVals_2
    
    _,W3=genetate_net_connectivity(p,IDENTICAL_HIER=1,ZERO_FLN=1,MACAQUE_CASE=0,STRONG_GBA=1,CONSENSUS_CASE=CONSENSUS_CASE)
    eigVals_3,_ = np.linalg.eig(W3)
    eigVals_3=np.sort(np.real(eigVals_3))
    tau_3=-3/eigVals_3
    
    fig,(ax1,ax2)=plt.subplots(2,1)
    ax1.plot(np.arange(2*p['n_area']),eigVals_1,'-o',label="original")
    ax1.plot(np.arange(2*p['n_area']),eigVals_2,'-o',label="disconnected")
    ax1.plot(np.arange(2*p['n_area']),eigVals_3,'-o',label="disconnected & identical area")
    ax1.legend()
    ax1.set_xlabel("eigenvalue index")
    ax1.set_ylabel("eigenvalue")
    
    ax2.plot(np.arange(2*p['n_area']),tau_1,'-o',label="original")
    ax2.plot(np.arange(2*p['n_area']),tau_2,'-o',label="disconnected")
    ax2.plot(np.arange(2*p['n_area']),tau_3,'-o',label="disconnected & identical area")
    ax2.legend()
    ax2.set_xlabel("time constant index")
    ax2.set_ylabel("time constant")

# identify hubs of a given connectivity matrix
def identify_hubs(W_t,per):
    W=W_t.copy()
    
    # fig,(ax1,ax2)=plt.subplots(1,2)
    # ax1.pcolormesh(W)
    
    size=np.size(W)
    thr_num=int(per*size)
    
    W_flat=W.flatten()
    W_sort=np.sort(-W_flat)
    thr=-W_sort[thr_num]
    W[W<thr]=0
    
    print('thr=',thr)
    
    # ax2.pcolormesh(W)
    
    G = nx.DiGraph() 
    
    for i in np.arange(W.shape[0]):
        for j in np.arange(W.shape[1]):
            if W[i,j]>0:
                G.add_edge(j,i)
                
    # nx.draw_networkx(G,with_labels=True)            
    bc=nx.betweenness_centrality(G) 
    return bc
    
# identify structural and functional hubs in a marmoset network 
def str_func_hub_comparison(p_t,W_s_t,W_f_t,per):
    p=p_t.copy()
    W_s=W_s_t.copy()
    W_f=W_f_t.copy()
    
    bc_struct = identify_hubs(W_s,per)
    bc_funct = identify_hubs(W_f,per)
    
    ref=np.array([45,45,45,45,45,34,45,45,45,33,29,45,45,18,13,45,11,15,43,45,45,20,45,31,45,45,45,3,26,37,45,8,9, 45, 38,45,45,42,45,6,28,45,12,25,2,15,45,16,45,1,35,10,45,45,45])

    ref_r=np.ones_like(ref)
    ref_r[ref<15]=2
    ref_r[ref<6]=2
    ref_r[ref<4]=3
    
    bc_s=np.zeros(p['n_area'])
    bc_f=np.zeros(p['n_area'])
    temp_bc_f=np.zeros(p['n_area'])
    ref_bc_f=np.zeros(p['n_area'])
    
    xtick_label=[]
    
    for i in np.arange(p['n_area']):
        temp_bc_f[i]=bc_funct[i]
        
    ind=np.argsort(-temp_bc_f)
    
    for i in np.arange(p['n_area']):
        bc_s[i]=bc_struct[ind[i]]
        bc_f[i]=bc_funct[ind[i]]
        ref_bc_f[i]=ref_r[ind[i]]
        xtick_label.append(p['areas'][ind[i]])
     
    ref_bc_f=ref_bc_f/np.max(ref_bc_f)*np.max(bc_f)    
    
    x = np.arange(p['n_area'])  # the label locations
    width = 0.3  # the width of the bars
        
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(x - width, bc_s, width, label='structure BC')
    ax.bar(x, bc_f, width, label='function BC')
    #ax.bar(x + width/2, ref_bc_f, width/2, label='ref function BC')
    
    ax.set_ylabel('betweeness centerality')
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_label)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.legend()
    fig.savefig('result/hist-BC.pdf')
    
    pe_corr=np.corrcoef(bc_s,bc_f)
    sp_corr=stats.spearmanr(bc_s,bc_f)
    
    print('sp_corr=',sp_corr[0])
    print('pe_corr=',pe_corr[0,1])
    
def scan_parameter_for_hub(W_f_t):
    W_f=W_f_t.copy()
    
    ref=np.array([45,45,45,45,45,34,45,45,45,33,29,45,45,18,13,45,11,15,43,45,45,20,45,31,45,45,45,3,26,37,45,8,9, 45, 38,45,45,42,45,6,28,45,12,25,2,15,45,16,45,1,35,10,45,45,45])
    ref_r=np.ones_like(ref)
    ref_r[ref<15]=4
    ref_r[ref<6]=6
    ref_r[ref<4]=8
    
    n_area=len(ref)
    j=0
    per_range=np.arange(0.6,0.2,-0.01)
    sp_corr=np.zeros(len(per_range))
    
    for per in per_range:
        bc_funct = identify_hubs(W_f,per)   
        bc_f=np.zeros(n_area)
        for k in np.arange(n_area):
            bc_f[k]=bc_funct[k]
        
        sp_corr[j],p_val=stats.spearmanr(ref_r,bc_f)
        j=j+1
        
    fig,ax=plt.subplots()
    ax.plot(per_range,sp_corr,'-o')
    ax.set_xlabel('threshold')
    ax.set_ylabel('spearman correlation')
    fig.savefig('result/BC_corr_vs_thre.pdf')

#------------------------------------------------------------------------------
#auxiliary functions used in other functions
#------------------------------------------------------------------------------
    
# plot the relation between the E and I parts of the eigenvectors
def plot_re_ri_eigvecs(p,eigVecs_t):
    eigVecs=eigVecs_t.copy()
    lens=p['n_area']
    fig,ax=plt.subplots()
    for i in np.arange(lens):
            ax.scatter(eigVecs[2*i,lens+1:-1],eigVecs[2*i+1,lens+1:-1],label=p['areas'][i])
    ax.legend()
    ax.set_xlabel('E population rate')
    ax.set_ylabel('I population rate')
    ax.legend(loc='best',ncol=2)

#define inverse participation ratio to meaure the non-spatial locality of a eigenvector    
def IPR(vec_t,ind=4):
    vec=vec_t.copy()
    vec=np.abs(vec)
    ipr_ind=np.sum(np.power(vec, ind)) / (np.power(np.sum(np.power(vec, 2)), 2)+2e-16)
    # ipr_ind=0
    # for i in range(len(vec)):
    #     ipr_ind=ipr_ind+vec[i]**ind
    return ipr_ind
 
def IPR_MATRIX(W_t):
    W=W_t.copy()
    r,c=W_t.shape
    ipr=np.zeros(c)
    for i in range(c):
        ipr[i]=IPR(W[:,i])
    mean_ipr=np.mean(ipr)
    return ipr,mean_ipr
    
#define spatial localization to meaure the spatial locality of a eigenvector    
def SL(vec_t,d_mat_t,ind=4):
    vec=vec_t.copy()
    d_mat=d_mat_t.copy()
    sl_ind=0
    lens=len(vec)
    for i in np.arange(lens):
        for j in np.arange(lens):
            sl_ind=sl_ind+vec[i]**ind * vec[j]**ind * np.exp(-d_mat[i,j]/np.mean(d_mat))
            #sl_ind=sl_ind+vec[i]**ind * vec[j]**ind * np.exp(-2*d_mat[i,j]/np.mean(d_mat))
    return sl_ind

#define theta to meaure the spatial locality of a eigenvector    
def THETA(vec_t,d_mat_t):
    vec=vec_t.copy()
    d_mat=d_mat_t.copy()
    theta=SL(vec,d_mat)/(IPR(vec)**2)
    return theta

def THETA_MATRIX(W_t,dist_mat_t):
    W=W_t.copy()
    dist_mat=dist_mat_t.copy()
    r,c=W_t.shape
    theta=np.zeros(c)
    for i in range(c):
        theta[i]=THETA(W[:,i],dist_mat)
    mean_theta=np.mean(theta)
    return theta,mean_theta

def local_index(x_ori,y_ori,x_new,y_new):
    ind=np.mean(x_new)/np.mean(x_ori)*np.mean(y_new)/np.mean(y_ori)
    return ind

#generate gaussian noise    
def gaussian_noise(mu,sigma,n_t):
    input_sig=np.zeros(n_t)
    for i in range(0,n_t):
        input_sig[i]=random.gauss(mu,sigma)
    return input_sig

def single_exp(x,a,b):
    return b*np.exp(-x/a)

def double_exp(x,a,b,c,d):
    return c*np.exp(-x/a)-d*np.exp(-x/b)

#compute eigen vector and value for desired direction.
def pick_eigen_direction(egval, egvec, direction_ref, mode, ord=0):
    # Return eigen vector and value for desired direction.
    # @param egval: eigen values list.
    # @param egvec: each column is an eigen vector, normalized to 2-norm == 1.
    # @param direction_ref: reference direction, can be a vector or column vectors.
    # @param mode:
    # @param ord: to select the `ord`-th close direction. For switching direction.
    if mode == 'smallest':
        id_egmin = np.argmin(np.abs(egval))
        zero_g_direction = egvec[:, id_egmin]
        if np.dot(zero_g_direction, direction_ref) < 0:
            zero_g_direction = -zero_g_direction
        return egval[id_egmin], zero_g_direction
    elif mode == 'continue':
        # pick the direction that matches the previous one
        if direction_ref.ndim==1:
            direction_ref = direction_ref[:, np.newaxis]
        n_pick = direction_ref.shape[1]    # number of track
        vec_pick = np.zeros_like(direction_ref)
        val_pick = np.zeros_like(egval)
        similarity = np.dot(egvec.conj().T, direction_ref)
        for id_v in range(n_pick):
            # id_pick = np.argmin(np.abs(np.abs(similarity[:, id_v])-1))
            id_pick = np.argsort(np.abs(np.abs(similarity[:, id_v])-1))[ord]
            if similarity[id_pick, id_v] > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            val_pick[id_v] = egval[id_pick]
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    elif mode == 'close-egval':
        # direction_ref should be pair (egval, egvec)
        old_egval = direction_ref[0]
        old_egvec = direction_ref[1]
        if len(old_egvec.shape) == 1:
            old_egvec = old_egvec[:, np.newaxis]
        egdist = np.abs(old_egval[:,np.newaxis] - egval[np.newaxis, :])
        # Greedy pick algo 1
        # 1. loop for columns of distance matrix
        # 2.    pick most close pair of eigenvalue and old-eigenvalue.
        # 3.    remove corresponding row in the distance matrix.
        # 4. go to 1.
        # Greedy pick algo 2
        # 1. pick most close pair of eigenvalue and old-eigenvalue.
        # 2. remove corresponding column and row in the distance matrix.
        # 3. go to 1.
        # Use algo 1.
        #plt.matshow(egdist)
        n_pick = old_egvec.shape[1]
        mask = np.arange(n_pick)
        vec_pick = np.zeros_like(old_egvec)
        val_pick = np.zeros_like(egval)
        #print('old_eigval=\n', old_egval[:,np.newaxis])
        #print('new eigval=\n', egval[:,np.newaxis])
        for id_v in range(n_pick):
            id_pick_masked = np.argmin(egdist[id_v, mask])
            #print('mask=',mask)
            id_pick = mask[id_pick_masked]
            #print('id_pick=',id_pick, '  eigval=', egval[id_pick])
            val_pick[id_v] = egval[id_pick]
            # might try: sign = np.exp(np.angle(...)*1j)
            if np.angle(np.vdot(egvec[:,id_pick], old_egvec[:, id_v])) > 0:
                vec_pick[:, id_v] = egvec[:, id_pick]
            else:
                vec_pick[:, id_v] = -egvec[:, id_pick]
            mask = np.delete(mask, id_pick_masked)
        return np.squeeze(val_pick), np.squeeze(vec_pick)
    else:
        raise ValueError()    
        

#compute the eigenvalues and eigenvectors of A+dA by perturbation analysis
def eig_approx(A,dA):
    
    eigval, eigvec = np.linalg.eig(A)
    id_sort = np.argsort(abs(eigval.real))
    eigval_ori=eigval[id_sort]
    eigvec_ori=eigvec[:, id_sort]
    
    eigval_l, eigvec_l = np.linalg.eig(A.T)
    id_sort_l = np.argsort(abs(eigval_l.real))
    eigvec_ori_l=eigvec_l[:, id_sort_l]
    
    eigvec_ori_left=eigvec_ori_l.conj()
    
    # print('eigval=',eigval_ori)
    # print('eigval_l=',eigval_ori_l)
    
    A_size=A.shape[0]
    d_eigval_1=np.zeros(A_size, dtype='complex128')
    d_eigval_2=np.zeros(A_size, dtype='complex128')
    d_eigvec_1=np.zeros((A_size,A_size), dtype='complex128')
    d_eigvec_2=np.zeros((A_size,A_size), dtype='complex128')

    for i in range(A_size):
        d_eigval_1[i]=eigvec_ori_left[:,i].conj().T@dA@eigvec_ori[:,i]/(eigvec_ori_left[:,i].conj().T@eigvec_ori[:,i])
        
        for k in range(A_size):
            if k!=i:
                d_eigvec_1[:,i]=d_eigvec_1[:,i]+eigvec_ori_left[:,k].conj().T@dA@eigvec_ori[:,i]*eigvec_ori[:,k]/(eigval_ori[i]-eigval_ori[k])/(eigvec_ori_left[:,k].conj().T@eigvec_ori[:,k])
                d_eigval_2[i]=d_eigval_2[i]+(eigvec_ori_left[:,k].conj().T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,i].conj().T@dA@eigvec_ori[:,k])/(eigval_ori[i]-eigval_ori[k])/(eigvec_ori_left[:,k].conj().T@eigvec_ori[:,k])/(eigvec_ori_left[:,i].conj().T@eigvec_ori[:,i])
    
        # for m in range(A_size):
        #     if m!=i:
        #         d_eigvec_2[:,i]=d_eigvec_2[:,i]+(eigvec_ori_left[:,i].T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,m].T@dA@eigvec_ori[:,i])/(eigval_ori[i]-eigval_ori[m])*eigvec_ori[:,m]/(eigval_ori[m]-eigval_ori[i])/(eigvec_ori_left[:,i].T@eigvec_ori[:,i])/(eigvec_ori_left[:,m].T@eigvec_ori[:,m])
        #         for k in range(A_size):
        #             if k!=i:
        #                 d_eigvec_2[:,i]=d_eigvec_2[:,i]-(eigvec_ori_left[:,k].T@dA@eigvec_ori[:,i])*(eigvec_ori_left[:,m].T@dA@eigvec_ori[:,k])/(eigval_ori[i]-eigval_ori[k])*eigvec_ori[:,m]/(eigval_ori[m]-eigval_ori[i])/(eigvec_ori_left[:,m].T@eigvec_ori[:,m])/(eigvec_ori_left[:,k].T@eigvec_ori[:,k])
  
    return eigval_ori+d_eigval_1, eigval_ori+d_eigval_1+d_eigval_2,eigvec_ori+d_eigvec_1,eigvec_ori+d_eigvec_1+d_eigvec_2

#compute the angle between two eigenvectors    
def normality_meaure(W_t):
    W=W_t.copy()
    n=np.shape(W)[1]
    
    for i in np.arange(n):
        W[:,i]=W[:,i]/np.linalg.norm(W[:,i])
        
    theta_mat=W.T@W
    theta_ang=np.zeros_like(theta_mat)
    for i in np.arange(n):
        for j in np.arange(n):
            if theta_mat[i,j]>1:
                theta_mat[i,j]=1
            theta_ang[i,j]=np.arccos(theta_mat[i,j])*180/np.pi
            theta_ang[i,j]=min([theta_ang[i,j], 180-theta_ang[i,j]])

    fig, ax = plt.subplots(figsize=(13,10))
    f=ax.pcolormesh(theta_ang,cmap='hot_r')
    fig.colorbar(f,ax=ax,pad=0.1)
    ax.set_title('theta angle')
    ax.invert_yaxis()
    
    mean_ang=(np.sum(theta_ang)-n*0)/(n**2-n)
    print('mean_ang=',mean_ang)

def matrix_random_permutation(p_t,FLN_t,SHUFFLE_TYPE=0,MACAQUE_CASE=1,CONSENSUS_CASE=0):
    
    W_shuffle=FLN_t.copy()
    temp_W=np.zeros_like(W_shuffle)
    # fig, ax=plt.subplots(1,3,figsize=(40,8))
    # vmin=np.min(W_shuffle)
    # vmax=np.max(W_shuffle)
    # f=ax[0].pcolor(FLN_t, cmap='bwr',norm=LogNorm(vmin=1e-7, vmax=1))
    # fig.colorbar(f,ax=ax[0],pad=0.15)
    # ax[0].set_title('original FLN matrix') 
    # ax[0].invert_yaxis()
    
    if SHUFFLE_TYPE<9:
        if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
            mask = np.eye(W_shuffle.shape[0], dtype=bool)
            print("SHUFFLE_TYPE==0:  all elements are randomly permuted")
        
        if  SHUFFLE_TYPE==1:  #only permute the nonzero elements
            mask = (W_shuffle==0)
            print("SHUFFLE_TYPE==1:  only permute the nonzero elements")
        
        if  SHUFFLE_TYPE==5:  #only permute the nonzero elements
            thr_fln = 0.05 # 0.0075
            mask = (W_shuffle < thr_fln)
            print("SHUFFLE_TYPE==5:  only permute the large elements")
        
        if  SHUFFLE_TYPE==2 or SHUFFLE_TYPE==3: 
            fln_flat=FLN_t.flatten()
            fln_flat=fln_flat[fln_flat>0]
            idx_fln=np.argsort(fln_flat)
            fln_sort=np.sort(fln_flat)
            thr_fln=fln_sort[int(len(idx_fln)*0.67)]  #0.67
            print('thr_fln=',thr_fln)
            if SHUFFLE_TYPE==2:  #only permute the small elements
                print("SHUFFLE_TYPE==2:  only permute the small elements")
                mask = (W_shuffle>thr_fln)
            else:                #only permute the large elements
                print("SHUFFLE_TYPE==3:  only permute the large elements")
                mask = (W_shuffle<thr_fln)
                 
            np.fill_diagonal(mask, True)

        if  SHUFFLE_TYPE==4:  #only permute the nonzero elements
            mask_zero = (W_shuffle==0)
            mask_corner = np.ones((p_t['n_area'], p_t['n_area']))
            mask_corner[-5:-1, :] = 0; mask_corner[:, -5:-1] = 0
            mask = np.logical_or(mask_zero, mask_corner)
            print("SHUFFLE_TYPE==4:  only permute the nonzero and last row& column elements")
            
        if SHUFFLE_TYPE==7:
            p=p_t.copy()
            mask_zero = (W_shuffle==0)
            k = 10
            W_neighbor = np.zeros((p['n_area'], p['n_area']))
            for i in range(k):
                W_neighbor += (np.diag(np.ones(p['n_area']-i,), -i) + np.diag(np.ones(p['n_area']-i,), i))
            mask_neighbor = W_neighbor > 0
            mask = np.logical_or(mask_zero, mask_neighbor)
            print("SHUFFLE_TYPE==7:  only permute the nonzero & non-neighbour elements")
            
        if SHUFFLE_TYPE==8:
            p=p_t.copy()
            mask_zero = (W_shuffle==0)
            k = 2
            W_neighbor = np.zeros((p['n_area'], p['n_area']))
            for i in range(k):
                W_neighbor += (np.diag(np.ones(p['n_area']-i,), -i) + np.diag(np.ones(p['n_area']-i,), i))
            mask_neighbor = np.logical_not(W_neighbor > 0)
            mask = np.logical_or(mask_zero, mask_neighbor)
            print("SHUFFLE_TYPE==8:  only permute the nonzero & neighbour elements")

            
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
            
        # if SHUFFLE_TYPE==4:   #make the smallest values be zero
        #     print("SHUFFLE_TYPE== 4:  make the smallest values be zero")
        #     W_shuffle[W_shuffle<thr_fln]=0
        
            
    # if SHUFFLE_TYPE==9:
    #     p=p_t.copy()
    #     mask = (W_shuffle==0)
        
    #     temp_W[~mask]=W_shuffle[~mask]
    #     element_buffer = W_shuffle[~mask]
                
    #     np.random.shuffle(element_buffer)
    #     W_shuffle[~mask] = element_buffer
        
    #     alpha = 10; eta = 5; mag = 2;
    #     dist_mat = p['dist_mat']
    #     dist_mat = dist_mat[0:p['n_area'], 0:p['n_area']]
    #     dist_normal_mask =  mag * (1 + alpha*np.exp(-dist_mat/eta)) / (1+alpha)
    #     W_shuffle = W_shuffle * dist_normal_mask
        
    #     print("SHUFFLE_TYPE==9:  only permute the nonzero elements & distance normalized")
        
    
    if SHUFFLE_TYPE==9:
        p=p_t.copy()
        fln_mat = FLN_t.copy()
        dist_mat = p['dist_mat']
        
        mask = dist_mat > np.max(dist_mat)/3
        
        conn_sym = p['conn_sym']
        conn_sym = np.logical_and(conn_sym, mask)
        fln_l = fln_mat[np.tril(conn_sym)]
        fln_u = (fln_mat.T)[np.tril(conn_sym)]
        permu_index = np.random.permutation(len(fln_l))
        fln_l_shuffle = fln_l[permu_index]
        fln_u_shuffle = fln_u[permu_index]
        W_shuffle = fln_mat.copy()
        W_shuffle[np.tril(conn_sym)] = fln_l_shuffle
        W_shuffle_t = (fln_mat.T).copy()
        W_shuffle_t[np.tril(conn_sym)] = fln_u_shuffle
        index_u = np.triu(np.ones([p_t['n_area'], p_t['n_area']]) > 0)
        W_shuffle[index_u] = (W_shuffle_t.T)[index_u]
        
        print("SHUFFLE_TYPE==9:  permute the distant connections (keep symmetry)")
        
    if SHUFFLE_TYPE==10:
        p=p_t.copy()
        dist_mat = p['dist_mat']
        mask = np.logical_or(dist_mat < np.max(dist_mat)/3, W_shuffle==0)
        
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
        
        print("SHUFFLE_TYPE==10:  only permute the distant connections (keep structure)")

        
    if SHUFFLE_TYPE==11:
        p=p_t.copy()
        dist_mat = p['dist_mat']
        mask = (dist_mat < np.max(dist_mat)/2)
        
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
        
        print("SHUFFLE_TYPE==11:  only permute the distant connections (random structure)")
        
    if SHUFFLE_TYPE==12:
        p=p_t.copy()
        dist_mat = p['dist_mat']
        mask = (dist_mat < np.max(dist_mat)/2)
        
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
        
        mask = (dist_mat > np.max(dist_mat)/3)
        
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
        
        print("SHUFFLE_TYPE==12:  permute the connections w.r.t distance (random structure)")
    
    if SHUFFLE_TYPE==13:
        p=p_t.copy()
        dist_mat = p['dist_mat']
        
        mask = np.logical_or(dist_mat > np.max(dist_mat)/3, W_shuffle==0)
        
        temp_W[~mask]=W_shuffle[~mask]
        element_buffer = W_shuffle[~mask]
                
        np.random.shuffle(element_buffer)
        W_shuffle[~mask] = element_buffer
        
        print("SHUFFLE_TYPE==13:  permute the close connections (keep structure)")
        
    if SHUFFLE_TYPE==14:
        p=p_t.copy()
        fln_mat = FLN_t.copy()
        dist_mat = p['dist_mat']
        
        mask = dist_mat < np.max(dist_mat)/3
        
        conn_sym = p['conn_sym']
        conn_sym = np.logical_and(conn_sym, mask)
        fln_l = fln_mat[np.tril(conn_sym)]
        fln_u = (fln_mat.T)[np.tril(conn_sym)]
        permu_index = np.random.permutation(len(fln_l))
        fln_l_shuffle = fln_l[permu_index]
        fln_u_shuffle = fln_u[permu_index]
        W_shuffle = fln_mat.copy()
        W_shuffle[np.tril(conn_sym)] = fln_l_shuffle
        W_shuffle_t = (fln_mat.T).copy()
        W_shuffle_t[np.tril(conn_sym)] = fln_u_shuffle
        index_u = np.triu(np.ones([p_t['n_area'], p_t['n_area']]) > 0)
        W_shuffle[index_u] = (W_shuffle_t.T)[index_u]
        
        print("SHUFFLE_TYPE==14:  permute the close connections (keep symmetry)")

    
        
    if SHUFFLE_TYPE==15:   #shuffle such that fln is anti-correlated with distance 
        print("SHUFFLE_TYPE==15: shuffle such that fln is anti-correlated with distance")
        p=p_t.copy()
        _,full_dist_mat = generate_dist_matrix(p,MACAQUE_CASE=MACAQUE_CASE,CONSENSUS_CASE=CONSENSUS_CASE)
        W_shuffle=sort_swap(full_dist_mat[:p['n_area'],:p['n_area']],W_shuffle,mask = ~np.eye(p['n_area'],dtype=bool))
    
    if SHUFFLE_TYPE==17:   #shuffle such that fln is anti-correlated with difference of hierachcy
        print("SHUFFLE_TYPE==17: shuffle such that fln is anti-correlated with delta hi")
        p=p_t.copy()
        hier_scale = p['exc_scale']
        diff_hier = hier_scale - np.reshape(hier_scale, (p['n_area'], 1))
        W_shuffle=sort_swap(np.abs(diff_hier),W_shuffle,mask = ~np.eye(p['n_area'],dtype=bool))
        
    if SHUFFLE_TYPE==16:   #make the smallest values be zero
        permu_index = np.random.permutation(p_t['n_area'])
        Wp = W_shuffle[permu_index, :]; 
        # permu_index = np.random.permutation(p_t['n_area'])
        Wp = Wp[:, permu_index];
        print("SHUFFLE_TYPE== 16:  permutate the hi")
        W_shuffle = Wp
    
    if SHUFFLE_TYPE==18:   #make the smallest values be zero
        permu_index = np.random.permutation(p_t['n_area'])
        Wp = W_shuffle[permu_index, :]; 
        print("SHUFFLE_TYPE== 18:  permutate the row")
        W_shuffle = Wp
        
    if SHUFFLE_TYPE==19:   #make the smallest values be zero
        permu_index = np.random.permutation(p_t['n_area'])
        Wp = W_shuffle[:, permu_index];
        print("SHUFFLE_TYPE== 19:  permutate the column")
        W_shuffle = Wp
        
    if SHUFFLE_TYPE==20:  # shuffle fln but keep the symmetricity
        # Need p has the parameter p['conn_sym']
        print("SHUFFLE_TYPE== 20:  symmetric permutation")
        p = p_t.copy()
        fln_mat = FLN_t.copy()
        conn_sym = p['conn_sym']
        fln_l = fln_mat[np.tril(conn_sym)]
        fln_u = (fln_mat.T)[np.tril(conn_sym)]
        permu_index = np.random.permutation(len(fln_l))
        fln_l_shuffle = fln_l[permu_index]
        fln_u_shuffle = fln_u[permu_index]
        W_shuffle = fln_mat.copy()
        W_shuffle[np.tril(conn_sym)] = fln_l_shuffle
        W_shuffle_t = (fln_mat.T).copy()
        W_shuffle_t[np.tril(conn_sym)] = fln_u_shuffle
        index_u = np.triu(np.ones([p_t['n_area'], p_t['n_area']]) > 0)
        W_shuffle[index_u] = (W_shuffle_t.T)[index_u]
        
        mask = np.logical_xor(p['conn'], p['conn_sym'])
        element_buffer = W_shuffle[mask]                
        np.random.shuffle(element_buffer)
        W_shuffle[mask] = element_buffer
        
    if SHUFFLE_TYPE==21:  # shuffle fln but keep the symmetricity
        # Need p has the parameter p['conn_sym']
        print("SHUFFLE_TYPE== 21:  non-symmetric permutation")
        p = p_t.copy()
        fln_mat = FLN_t.copy()
        conn_sym = p['conn_sym']
        fln_l = fln_mat[np.tril(conn_sym)]
        fln_u = (fln_mat.T)[np.tril(conn_sym)]
        permu_index = np.random.permutation(len(fln_l))
        fln_l_shuffle = fln_l[permu_index]
        permu_index2 = np.random.permutation(len(fln_l))
        fln_u_shuffle = fln_u[permu_index2]
        W_shuffle = fln_mat.copy()
        W_shuffle[np.tril(conn_sym)] = fln_l_shuffle
        W_shuffle_t = (fln_mat.T).copy()
        W_shuffle_t[np.tril(conn_sym)] = fln_u_shuffle
        index_u = np.triu(np.ones([p_t['n_area'], p_t['n_area']]) > 0)
        W_shuffle[index_u] = (W_shuffle_t.T)[index_u]
        
        mask = np.logical_xor(p['conn'], p['conn_sym'])
        element_buffer = W_shuffle[mask]                
        np.random.shuffle(element_buffer)
        W_shuffle[mask] = element_buffer
        
    if SHUFFLE_TYPE==22:  # shuffle fln but keep the symmetricity
        # Need p has the parameter p['conn_sym']
        print("SHUFFLE_TYPE== 22:  symmetric permutation")
        p = p_t.copy()
        fln_mat = FLN_t.copy()
        conn_sym = p['conn_sym']
        fln_l = fln_mat[np.tril(conn_sym)]
        fln_u = (fln_mat.T)[np.tril(conn_sym)]
        permu_index = np.random.permutation(len(fln_l))
        fln_l_shuffle = fln_l[permu_index]
        permu_index2 = np.random.permutation(len(fln_l))
        fln_u_shuffle = fln_u[permu_index2]
        # fln_u_shuffle = fln_u[permu_index]
        W_shuffle = fln_mat.copy()
        W_shuffle[np.tril(conn_sym)] = fln_u_shuffle
        W_shuffle_t = (fln_mat.T).copy()
        W_shuffle_t[np.tril(conn_sym)] = fln_l_shuffle
        index_u = np.triu(np.ones([p_t['n_area'], p_t['n_area']]) > 0)
        W_shuffle[index_u] = (W_shuffle_t.T)[index_u]
        
        mask = np.logical_xor(p['conn'], p['conn_sym'])
        element_buffer = W_shuffle[mask]                
        np.random.shuffle(element_buffer)
        W_shuffle[mask] = element_buffer
    #     fig3,ax3=plt.subplots()
    #     ax3.scatter(full_dist_mat[:p['n_area'],:p['n_area']].flatten(),W_shuffle.flatten())
    #     ax3.set_xlabel('distance')
    #     ax3.set_ylabel('FLN')
    
    # if SHUFFLE_TYPE==6:  #shuffle such that fln is anti-correlated with hierarchy difference
    #     print("SHUFFLE_TYPE==6: shuffle such that fln is anti-correlated with hierarchy difference")
    #     p=p_t.copy()
    #     hier_mat=np.zeros((p['n_area'],p['n_area']))
    #     for i in np.arange(p['n_area']):
    #         for j in np.arange(p['n_area']):
    #             hier_mat[i,j]=np.abs(p['hier_vals'][i]-p['hier_vals'][j])
    #     W_shuffle=sort_swap(hier_mat,W_shuffle,mask = ~np.eye(p['n_area'],dtype=bool))
    #     fig3,ax3=plt.subplots()
    #     ax3.scatter(hier_mat.flatten(),W_shuffle.flatten())
    #     ax3.set_xlabel('hierarchy difference')
    #     ax3.set_ylabel('FLN')
        
    # #---------------plot the original and shuffled matrices--------------------
    # f1=ax[1].pcolor(temp_W, cmap='bwr',norm=LogNorm(vmin=1e-7, vmax=1))
    # fig.colorbar(f1,ax=ax[1],pad=0.15)
    # ax[1].set_title('to be shuffled FLN matrix') 
    # ax[1].invert_yaxis()
        
    # f2=ax[2].pcolor(W_shuffle, cmap='bwr',norm=LogNorm(vmin=1e-7, vmax=1))
    # fig.colorbar(f2,ax=ax[2],pad=0.15)
    # ax[2].set_title('shuffled FLN matrix') 
    # ax[2].invert_yaxis()
    # plt.close(fig)    

    return W_shuffle
    
def unstability_detection(p_t,fln_t):
    
    p=p_t.copy()
    fln=fln_t.copy()
    
    p['exc_scale'] = (1+p['eta']*p['hier_vals'])
    p['local_exc_scale'] = (1+p['eta_local']*p['hier_vals'])
    p['inh_scale'] = (1+p['eta_inh']*p['hier_vals_inh'])
    p['local_inh_scale'] = (1+p['eta_inh_local']*p['hier_vals'])
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)

    # local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    # local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    # local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    # local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        
    fln_scaled = (p['exc_scale'] * fln.T).T
    fln_scaled_inh = (p['inh_scale'] * fln.T).T
    
    #---------------------------------------------------------------------------------
    # the first way to compute the connectivity matrix
    #---------------------------------------------------------------------------------
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):
        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

        W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
        W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled_inh[i,:]/p['tau_inh']
        
    eigvals, eigvecs = np.linalg.eig(W)   
    
    max_eigval=np.max(eigvals)
    
    return max_eigval, W

def time_constant_localization_shuffle_fln(p_t,FLN_shuffled_t,figname='time_scale_matrix_original.png'):
    
    p=p_t.copy()
    FLN_shuffled=FLN_shuffled_t.copy()
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        
    fln_scaled = (p['exc_scale'] * FLN_shuffled.T).T
    
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):
        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

        W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
        W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled[i,:]/p['tau_inh']
        
    W_EI=np.zeros_like(W)
    W_EI[0:p['n_area'],0:p['n_area']]=W.copy()[0::2,0::2]
    W_EI[0:p['n_area'],p['n_area']:]=W.copy()[0::2,1::2]
    W_EI[p['n_area']:,0:p['n_area']]=W.copy()[1::2,0::2]
    W_EI[p['n_area']:,p['n_area']:]=W.copy()[1::2,1::2]

    #---------------------------------------------------------------------------------
    # eigenmode decomposition
    #--------------------------------------------------------------------------------- 
    eigVals, eigVecs = np.linalg.eig(W_EI)    
    eigVecs_a=np.abs(eigVecs)
    
    tau=-1/np.real(eigVals)
    tau_s=np.zeros_like(tau)
    for i in range(len(tau)):
        tau_s[i]=format(tau[i],'.2f')
    
           
    area_name_list=p['areas']
    inv_eigVecs=np.linalg.inv(eigVecs)
    
    n=len(area_name_list)
    coef_green=np.zeros((n,2*n))+0j  #cofficient of the green's function 

    for i in np.arange(n):
        for j in np.arange(2*n):
            coef_green[i,j]=eigVecs[i,j]*inv_eigVecs[j,i]
                    
    ind=np.argsort(-tau_s)
    coef_green_reorder=np.zeros((p['n_area'],2*p['n_area']))+0j
    eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))
    eigVecs_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
    tau_reorder=np.zeros(2*p['n_area'])
    
    for i in range(2*p['n_area']):
        coef_green_reorder[:,i]=coef_green[:,ind[i]]
        tau_reorder[i]=tau_s[ind[i]]
        eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
        eigVecs_reorder[:,i]=eigVecs[:,ind[i]]
        tau_reorder[i]=tau_s[ind[i]]
    
    eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
    tau_slow=tau_reorder[:p['n_area']]
    
    coef_green_slow=coef_green_reorder[:,:p['n_area']]
    coef_normed_green=normalize_matrix(coef_green_slow,column=0)  #normailzied by row
    eigVecs_slow_normed=normalize_matrix(eigVecs_slow,column=1) #normalized by column
    
    eigVecs_slow_normed=np.fliplr(eigVecs_slow_normed)  #flip the IPRs of each area starts from V1, but no need to flip green because green func is sorted by row, which naturally starts from V1
    
    ipr_eigenvecs=np.zeros(p['n_area'])
    ipr_green=np.zeros(p['n_area'])

    #normalize the coefficient row by row
    for j in range(p['n_area']):
        ipr_eigenvecs[j]=IPR(np.abs(eigVecs_slow_normed[:,j])) #normailzied by column
        ipr_green[j]=IPR(np.abs(coef_normed_green[j,:])) #normalized by row
        
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(30,10))
    
    f=ax1.pcolormesh(eigVecs_slow_normed,cmap='hot')
    fig.colorbar(f,ax=ax1,pad=0.15)
    ax1.set_title('eigen matrix of slow mode')
    
    ax1.invert_yaxis()
    
    x = np.arange(len(tau_slow)) # xticks
    y = np.arange(p['n_area']) # yticks
    xlim = (0,len(tau_slow))
    ylim = (0,p['n_area'])
    
    yticklabels_odd=p['areas'][1::2]
    yticklabels_even=p['areas'][::2]
    
    # set original ticks and ticklabels
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_xticks(x[::5])
       
    ax1.set_xticklabels(tau_slow[-1::-5])
    ax1.set_yticks(y[::2])
    ax1.set_yticklabels(yticklabels_even)
    ax1.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax1.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax1.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()   
    
    f=ax2.pcolormesh(np.abs(coef_normed_green),cmap='hot')
    fig.colorbar(f,ax=ax2,pad=0.1)
    ax2.set_title('coef matrix of Greens func')
    
    ax2.invert_yaxis()
    ax2.invert_xaxis()

    # set original ticks and ticklabels
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_xticks(x[::5])
    ax2.invert_xaxis()
       
    ax2.set_xticklabels(tau_slow[::5])
    ax2.set_yticks(y[::2])
    ax2.set_yticklabels(yticklabels_even)
    ax2.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax4 = ax2.twinx()
    ax4.set_ylim(ylim)
    ax4.set_yticks(y[1::2])
    ax4.set_yticklabels(yticklabels_odd)
    ax4.invert_yaxis()   
    
    fig.savefig('result/'+figname)   
    plt.close(fig)
    
    return ipr_eigenvecs,ipr_green

def sigma_lambda_eigvector_localization_shuffle_fln(p_t,FLN_shuffled_t,MACAQUE_CASE=1):
    
    p=p_t.copy()
    FLN_shuffled=FLN_shuffled_t.copy()
    
    local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
        
    fln_scaled = (p['exc_scale'] * FLN_shuffled.T).T
    
    W=np.zeros((2*p['n_area'],2*p['n_area']))       
    
    for i in range(p['n_area']):
        W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
        W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
        W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
        W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

        W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
        W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled[i,:]/p['tau_inh']
        
    Sigma,Lambda=get_Sigma_Lambda_matrix(p,W,MACAQUE_CASE=MACAQUE_CASE)

    eigvals_norder,eigVecs_norder_sigma = np.linalg.eig(Sigma+Lambda)
    
    le=len(p['hier_vals'])
    id_mat=np.eye(le)
    ind=np.argsort(-np.real(eigvals_norder))
    per_mat=id_mat.copy()
    for i in np.arange(le):
        per_mat[i,:]=id_mat.copy()[ind[i],:]
    
    eigvecs_order_sigma=eigVecs_norder_sigma@(np.linalg.inv(per_mat))  
    
    _,mean_ipr=IPR_MATRIX(eigvecs_order_sigma)
    # print('mean_ipr=',mean_ipr)

    return mean_ipr
    
    
    
def sort_swap(A,B, mask = None):
    """Swap masked element between two array with sorted order.
    :param A: first input array
    :type A: 2d numpy array
    :param B: second input arrary
    :type B: 2d numpy array
    :param mask: mask array
    :type mask: 2d numpy array of bool
    :return: two swapped array
    :rtype: numpy array

    """
    if mask is None:
        # sorted all elements in matrix, return indices of row and col
        A_index = np.unravel_index(np.argsort(A, axis=None), A.shape)
        B_index = np.unravel_index(np.argsort(B, axis=None), B.shape)
        # flip the order of second sort indices
        B_index_flip = tuple(np.flip(B_index))
        # create copy
        A_new = A.copy()
        B_new = B.copy()
        # swap
        A_new[A_index] = B[B_index_flip]
        B_new[B_index_flip] = A[A_index]
    else:
        # create buffer for masked elements, 1d array here
        A_masked = A[mask]
        B_masked = B[mask]
        # sort 1d masked elements
        A_index = np.argsort(A_masked)
        B_index = np.argsort(B_masked)
        B_index_flip = np.flip(B_index)
        A_masked_new = A_masked.copy()
        B_masked_new = B_masked.copy()
        # swap masked copy
        A_masked_new[A_index] = B_masked[B_index_flip]
        B_masked_new[B_index_flip] = A_masked[A_index]
        A_new = A.copy()
        B_new = B.copy()
        # update original matrix
        A_new[mask] = A_masked_new
        B_new[mask] = B_masked_new
    return A_new    
    
    

def normalize_matrix(M_t,column=1):
    M=M_t.copy()
    r,c=M.shape
    M_norm=np.zeros_like(M)

    if column==1:
        for i in np.arange(c):
            M_norm[:,i]=M[:,i]/np.linalg.norm(M[:,i])
    else:
        for i in np.arange(r):
            M_norm[i,:]=M[i,:]/np.linalg.norm(M[i,:])
   
    return M_norm

def get_Sigma_Lambda_matrix(p_t,W_t,MACAQUE_CASE,CONSENSUS_CASE=0):
    
    p=p_t.copy()
    _,W0=genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=1,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
    #---------------------------------------------------------------------------------
    #reshape the connectivity matrix by E and I population blocks, EE, EI, IE, II
    #---------------------------------------------------------------------------------
    W0_EI=np.zeros_like(W0)
    W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
    W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
    W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
    W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
    
    W1=W_t.copy()
    W1_EI=np.zeros_like(W1)
    W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
    W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
    W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
    W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
    
    #the variable names are consistent with symbols used in the notes
    D=W0_EI
    F=W1_EI-W0_EI
    
    D_EE=W0_EI[0:p['n_area'],0:p['n_area']]
    D_IE=W0_EI[p['n_area']:,0:p['n_area']]
    D_EI=W0_EI[0:p['n_area'],p['n_area']:]
    D_II=W0_EI[p['n_area']:,p['n_area']:]
    
    # F_EE=F[0:p['n_area'],0:p['n_area']]
    # F_IE=F[p['n_area']:,0:p['n_area']]
        
    #--------------------------------------------------------------------------
    #approximations of A and B (see notes for detailed derivations)
    #--------------------------------------------------------------------------
    A=np.zeros_like(D_EE)
    A_app=np.zeros_like(A)
    B=np.zeros_like(A)
    B_app=np.zeros_like(A)
    
    for i in np.arange(p['n_area']):
        A[i,i]=0.5/D_IE[i,i]*(D_II[i,i]-D_EE[i,i]+np.sqrt((D_EE[i,i]+D_II[i,i])**2-4*(D_EE[i,i]*D_II[i,i]-D_EI[i,i]*D_IE[i,i])))
        A_app[i,i]=-D_EI[i,i]/D_II[i,i]
        B[i,i]=-D_IE[i,i]/(D_EE[i,i]+2*D_IE[i,i]*A[i,i]-D_II[i,i])
        B_app[i,i]=D_IE[i,i]/D_II[i,i]
         
  
    #--------------------------------------------------------------------------
    #compute P to diagnalize the local connectivity matrix without long-range connectivity
    #--------------------------------------------------------------------------
    P=np.zeros((2*p['n_area'],2*p['n_area']))
    P[0:p['n_area'],0:p['n_area']]=np.eye(p['n_area'])
    P[0:p['n_area'],p['n_area']:]=A
    P[p['n_area']:,0:p['n_area']]=B
    P[p['n_area']:,p['n_area']:]=np.eye(p['n_area'])+A@B
    P_inv=np.linalg.inv(P)
       
    #--------------------------------------------------------------------------
    #similarity transform on the connectivity matrix using P
    #--------------------------------------------------------------------------
    Lambda=P@D@P_inv
    Sigma=P@F@P_inv
    Lambda[np.abs(Lambda)<1e-12]=0
    Sigma[np.abs(Sigma)<1e-12]=0
    
    #--------------------------------------------------------------------------
    #extract block matrices after similarity transformation on the connectivity matrix
    #--------------------------------------------------------------------------
    Sigma_1=Sigma[0:p['n_area'],0:p['n_area']]
    # Sigma_2=Sigma[0:p['n_area'],p['n_area']:]
    # Sigma_3=Sigma[p['n_area']:,0:p['n_area']]
    # Sigma_4=Sigma[p['n_area']:,p['n_area']:]
    Lambda_1=Lambda[0:p['n_area'],0:p['n_area']]
    # Lambda_4=Lambda[p['n_area']:,p['n_area']:]
    
    return Sigma_1, Lambda_1
    
#------------------------------------------------------------------------------
#reproducing Rishi's result in Neuron 2015
#------------------------------------------------------------------------------    
#generate functional connectivity 
def generate_func_connectivity(p_t,W_t):
    p=p_t.copy()
    W=W_t.copy()
    eigVals, eigVecs = np.linalg.eig(W)
    Lambda = np.diag(eigVals)
    #---------------------------------------------------------------------------------
    # Check
    #--------------------------------------------------------------------------------- 
    inv_eigVecs=np.linalg.inv(eigVecs)
    Test=eigVecs.dot(Lambda).dot(inv_eigVecs)
    # fig, ax = plt.subplots()
    # f=ax.pcolormesh(np.real(Test)-W,cmap='hot')
    # fig.colorbar(f,ax=ax,pad=0.15)
    
    #---------------------------------------------------------------------------------
    # analytical functional connectivity
    #--------------------------------------------------------------------------------- 
    sigma=1e-5
    B=np.zeros_like(W)
    
    for i in np.arange(p['n_area']):
        B[2*i,2*i]=sigma
    
    Q=inv_eigVecs.dot(B).dot(B.conj().T).dot(inv_eigVecs.conj().T)
    M=np.zeros_like(Q)
    for i in np.arange(2*p['n_area']):
        for j in np.arange(2*p['n_area']):
            M[i,j]=-Q[i,j]/(eigVals[i]+eigVals[j].conj())
    
    Cov_mat=eigVecs.dot(M).dot(eigVecs.conj().T)
    Cov_mat_E = Cov_mat[0::2, 0::2]
    Corr_mat_E=np.zeros((p['n_area'],p['n_area']))
    for i in np.arange(p['n_area']):
        for j in np.arange(p['n_area']):
            Corr_mat_E[i,j]=np.real(Cov_mat[2*i,2*j])/np.sqrt(Cov_mat[2*i,2*i]*Cov_mat[2*j,2*j])      
    return Corr_mat_E, Cov_mat_E

#plot functional connectivity
def plot_func_connectivity(p_t,Corr_mat_E):
    
    p=p_t.copy()
    
    Corr_mat_cut=Corr_mat_E.copy()
    # Corr_mat_cut[Corr_mat_E>0.3]=0.3
    # Corr_mat_cut[Corr_mat_E<0.01]=0
    
    x = np.arange(len(p['areas'])) # xticks
    y = np.arange(len(p['areas'])) # yticks
    xlim = (0,len(p['areas']))
    ylim = (0,len(p['areas']))
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    fig, ax = plt.subplots()
    f=ax.pcolormesh(Corr_mat_cut,cmap='hot')
    ax.set_title('functional connectivity')    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(xticklabels_even)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()
    
    fig.colorbar(f,ax=ax,pad=0.15)
    fig.savefig('result/Functional_Connectivity.pdf')    

    #---------------------------------------------------------------------------------
    # correlation between functional connectivity and FLN
    #--------------------------------------------------------------------------------- 
    # FLN=p['fln_mat'] * p['sln_mat']
    FLN=p['fln_mat']        
    np.fill_diagonal(Corr_mat_E,0)
    np.fill_diagonal(FLN,0)
    corr_flat=Corr_mat_E.flatten()
    
    # FLN = FLN * p['sln_mat']
    fln_flat=FLN.flatten()
    # corr_flat[np.where(fln_flat==0)]=1e-8
    # fln_flat[np.where(fln_flat==0)]=1e-8
    
    # pick_index = (fln_flat > 1e-6) * (corr_flat > 1e-4)
    # fln_flat = fln_flat[pick_index]
    # corr_flat= corr_flat[pick_index]
    
    fig,ax=plt.subplots()
    ax.scatter(fln_flat,corr_flat)      
    ax.set_xlim((1e-6,1e0))
    ax.set_ylim((1e-4,1e0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLN')
    ax.set_ylabel('Functional Connectivity')
    
    ce=np.corrcoef(fln_flat,corr_flat)
    ax.set_title('corrcoef='+str(ce[0,1])[:4])
    print('corrcoef=',ce[0,1])
    fig.savefig('result/Coeff_VS_FLN.pdf')    

#plot structural connectivity in log scale
def plot_struct_connectivity_logscale(p_t):
    p=p_t.copy()
    
    np.fill_diagonal(p['fln_mat'],1e-10)
    
    x = np.arange(len(p['areas'])) # xticks
    y = np.arange(len(p['areas'])) # yticks
    xlim = (0,len(p['areas']))
    ylim = (0,len(p['areas']))
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    inferno_r = cm.get_cmap('inferno_r', 1024)
    newcolors = inferno_r(np.linspace(0, 1, 1024))
    white = np.array([1, 1, 1, 1]) #[R,G,B,alpha] ranging from 0 to 1
    newcolors[:1, :] = white       # set first color to black
    newcmp = ListedColormap(newcolors)

    fig, ax = plt.subplots()
    f=ax.pcolormesh(p['fln_mat'],cmap=newcmp,norm=LogNorm(vmin=1e-6, vmax=1))
    ax.set_title('structure connectivity (FLN)')    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(xticklabels_even)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()
    
    fig.colorbar(f,ax=ax,pad=0.15)
    fig.savefig('result/Structure_Connectivity.pdf')    
    
#plot functional connectivity in log scale    
def plot_func_connectivity_logscale(p_t,Corr_mat_E):
    
    p=p_t.copy()
    thr=1e-4
    Corr_mat_cut=Corr_mat_E.copy()
    #Corr_mat_cut[Corr_mat_E>0.3]=0.3
    Corr_mat_cut[Corr_mat_E<thr]=1e-4
    np.fill_diagonal(Corr_mat_cut,1e-4)
    
    x = np.arange(len(p['areas'])) # xticks
    y = np.arange(len(p['areas'])) # yticks
    xlim = (0,len(p['areas']))
    ylim = (0,len(p['areas']))
    
    xticklabels_odd  = p['areas'][1::2]
    xticklabels_even = p['areas'][::2]
    yticklabels_odd=xticklabels_odd
    yticklabels_even=xticklabels_even
    
    inferno_r = cm.get_cmap('inferno_r', 1024)
    newcolors = inferno_r(np.linspace(0, 1, 1024))
    white = np.array([1, 1, 1, 1]) #[R,G,B,alpha] ranging from 0 to 1
    newcolors[:1, :] = white       # set first color to black
    newcmp = ListedColormap(newcolors)

    fig, ax = plt.subplots()
    f=ax.pcolormesh(Corr_mat_cut,cmap=newcmp,norm=LogNorm(vmin=1e-4, vmax=1))
    ax.set_title('functional connectivity')    
    # set original ticks and ticklabels
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(x[::2])
    ax.set_xticklabels(xticklabels_even)
    ax.set_yticks(y[::2])
    ax.set_yticklabels(yticklabels_even)
    ax.invert_yaxis()
    # rotate xticklabels to 90 degree
    plt.setp(ax.get_xticklabels(), rotation=90)
    
    # second x axis
    ax2 = ax.twiny()
    ax2.set_xlim(xlim)
    ax2.set_xticks(x[1::2])
    ax2.set_xticklabels(xticklabels_odd)
    # rotate xticklabels to 90 degree
    plt.setp(ax2.get_xticklabels(), rotation=90)
    
    # second y axis
    ax3 = ax.twinx()
    ax3.set_ylim(ylim)
    ax3.set_yticks(y[1::2])
    ax3.set_yticklabels(yticklabels_odd)
    ax3.invert_yaxis()
    
    fig.colorbar(f,ax=ax,pad=0.15)
    fig.savefig('result/Functional_Connectivity.pdf')    

    #---------------------------------------------------------------------------------
    # correlation between functional connectivity and FLN
    #--------------------------------------------------------------------------------- 
    FLN=p['fln_mat']
            
    np.fill_diagonal(Corr_mat_E,0)
    np.fill_diagonal(FLN,0)
    corr_flat=Corr_mat_E.flatten()
    fln_flat=FLN.flatten()
    # corr_flat[np.where(fln_flat==0)]=1e-8
    # fln_flat[np.where(fln_flat==0)]=1e-8
    
    fig,ax=plt.subplots()
    ax.scatter(fln_flat,corr_flat)      
    ax.set_xlim((1e-6,1e0))
    ax.set_ylim((1e-4,1e0))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('FLN')
    ax.set_ylabel('Functional Connectivity')
    ce=np.corrcoef(fln_flat,corr_flat)
    ax.set_title('corrcoef='+str(ce[0,1])[:4])
    print('corrcoef=',ce[0,1])
    fig.savefig('result/Coeff_VS_FLN.pdf')    
    
#simulate macaque network and measure the time costant
#the time constant is defined as the decrease of response 5% above the baseline given a step current            
def run_stimulus_pulse_macaque(p_t,fln_mat):
    
    #---------------------------------------------------------------------------------
    # Redefine Parameters
    #---------------------------------------------------------------------------------

    p=p_t.copy()

    # Definition of combined parameters

    local_EE =  p['beta_exc'] * p['wEE'] * p['exc_scale']
    local_EI = -p['beta_exc'] * p['wEI'] * np.ones_like(local_EE)
    local_IE =  p['beta_inh'] * p['wIE'] * p['exc_scale']
    local_II = -p['beta_inh'] * p['wII'] * np.ones_like(local_EE)

    fln_scaled = (p['exc_scale'] * fln_mat.T).T
    
    #---------------------------------------------------------------------------------
    # Simulation Parameters
    #---------------------------------------------------------------------------------

    dt = 0.2   # ms
    T = 12500
    
    t_plot = np.linspace(0, T, int(T/dt)+1)
    n_t = len(t_plot)  

    # From target background firing inverts background inputs
    r_exc_tgt = 10 * np.ones(p['n_area'])    
    r_inh_tgt = 35 * np.ones(p['n_area'])

    longrange_E = np.dot(fln_scaled,r_exc_tgt)
    I_bkg_exc = r_exc_tgt - (local_EE*r_exc_tgt + local_EI*r_inh_tgt
                             + p['beta_exc']*p['muEE']*longrange_E)
    I_bkg_inh = r_inh_tgt - (local_IE*r_exc_tgt + local_II*r_inh_tgt
                             + p['beta_inh']*p['muIE']*longrange_E)

    # Set stimulus input
    I_stim_exc = np.zeros((n_t,p['n_area']))
 
    time_idx = (t_plot>0) & (t_plot<=250)
    # I_stim_exc[time_idx, area_stim_idx] = 41.187
    I_stim_exc[time_idx, :] = 50
    #---------------------------------------------------------------------------------
    # Storage
    #---------------------------------------------------------------------------------

    r_exc = np.zeros((n_t,p['n_area']))
    r_inh = np.zeros((n_t,p['n_area']))

    #---------------------------------------------------------------------------------
    # Initialization
    #---------------------------------------------------------------------------------
    fI = lambda x : x*(x>0)
    #fI = lambda x : x*(x>0)*(x<300)+300*(x>300)
    
    # Set activity to background firing
    r_exc[0] = r_exc_tgt
    r_inh[0] = r_inh_tgt
    
    tau_decay = np.zeros(p['n_area'])
    #---------------------------------------------------------------------------------
    # Running the network
    #---------------------------------------------------------------------------------

    for i_t in range(1, n_t):
        longrange_E = np.dot(fln_scaled,r_exc[i_t-1])
        I_exc = (local_EE*r_exc[i_t-1] + local_EI*r_inh[i_t-1] +
                 p['beta_exc'] * p['muEE'] * longrange_E +
                 I_bkg_exc + I_stim_exc[i_t])

        I_inh = (local_IE*r_exc[i_t-1] + local_II*r_inh[i_t-1] +
                 p['beta_inh'] * p['muIE'] * longrange_E + I_bkg_inh)
        
        d_r_exc = -r_exc[i_t-1] + fI(I_exc)
        d_r_inh = -r_inh[i_t-1] + fI(I_inh)

        r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt/p['tau_exc']
        r_inh[i_t] = r_inh[i_t-1] + d_r_inh * dt/p['tau_inh']
        
    for i_area in np.arange(p['n_area']):
        base_val=r_exc[:,i_area].min()
        max_val=r_exc[int(250/dt),i_area]-base_val
        
        p_end=np.where(r_exc[:,i_area]>0.05*max_val+base_val)[0][-1]
        tau_decay[i_area]=p_end*dt-250
    
    # #---------------------------------------------------------------------------------
    # # Plotting step input results
    # #---------------------------------------------------------------------------------
    
    # area_name_list = [p['areas'][area_stim_idx],p['areas'][area_stim_idx]]
       
    # area_idx_list=[-1]
    # for name in area_name_list:
    #     area_idx_list=area_idx_list+[p['areas'].index(name)]
    # #area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
    
    # f, ax_list = plt.subplots(len(area_idx_list), sharex=True)
    
    # clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
    # c_color=0
    # for ax, area_idx in zip(ax_list, area_idx_list):
    #     if area_idx < 0:
    #         y_plot = I_stim_exc[:, area_stim_idx].copy()
    #         z_plot = np.zeros_like(y_plot)
    #         txt = 'Input'

    #     else:
    #         y_plot = r_exc[:,area_idx].copy()
    #         z_plot = r_inh[:,area_idx].copy()
    #         txt = p['areas'][area_idx]

   
    #     y_plot = y_plot - y_plot.min()
    #     z_plot = z_plot - z_plot.min()
        
    #     ax.plot(t_plot, y_plot,color=clist[0][c_color])
    #     ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
    #     c_color=c_color+1
    #     ax.text(0.9, 0.6, txt, transform=ax.transAxes)

    #     ax.set_yticks([0,y_plot.max(),z_plot.max()])
    #     ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
    #     ax.spines["right"].set_visible(False)
    #     ax.spines["top"].set_visible(False)
    #     #ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')

    # f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
    # ax.set_xlabel('Time (ms)')    

    
    return tau_decay    
    