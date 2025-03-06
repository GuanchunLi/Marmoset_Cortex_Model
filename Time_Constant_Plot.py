# -*- coding: utf-8 -*-
"""
From Spyder Editor

Author: Guanchun Li (NYU Courant)

Time: Thu Jan 13 09:38:39 2022
"""


import sys, os
import numpy as np
import numpy.ma as ma
import scipy.io
import neurodsp as ndsp
from fooof import FOOOFGroup
from neurodsp import spectral
from fooof import FOOOF
import pandas as pd

from sklearn.manifold import SpectralEmbedding
from scipy import stats

# from mapalign import dist, embed
import pre_functions_clean as pf
import time_constant_shuffle_FLN as tc
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


def convert_knee_val(knee, exponent=2.):

    knee_freq = knee**(1./exponent)
    knee_tau = 1./(2*np.pi*knee_freq)
    return knee_freq, knee_tau

def compute_psds(data, fs=1000.):
    nperseg, noverlap, f_lim, spg_outlier_pct = int(fs), int(fs/2), 200., 5
    f_axis, psd_mean = spectral.compute_spectrum(data, fs, avg_type='median', nperseg=nperseg, noverlap=noverlap, f_range=[0, f_lim]) # , outlier_percent=spg_outlier_pct)
    # f_axis, psd_mean = spectral.compute_spectrum(data, fs, avg_type='median', nperseg=fs*2)
    psds_reg = np.log10(psd_mean)
    return f_axis, psds_reg

def compute_tau(f_axis, psds_reg, fit_range=[0,100],max_n_peaks=2, fs=1000.):
    # plt_inds = np.arange(fit_range[0],fit_range[1]+1)
    fok = FOOOF(max_n_peaks=max_n_peaks, aperiodic_mode='knee', verbose=False)
    fok.fit(f_axis, 10**psds_reg, fit_range)
    offset, knee, exp = fok.get_params('aperiodic_params')
    kfreq, tau = convert_knee_val(knee,exp)
    tau = fs*tau
    # ap_spectrum = np.log10((10**offset/(knee+f_axis**exp)))
    return tau, fok

def compute_time_constant(data, area, AOI):
    n_AOI = len(AOI)
    tau_lst = np.zeros([n_AOI,])
    fok_lst = [0] * n_AOI
    f_axis, psds_reg = compute_psds(data)
    for i in range(n_AOI):
        psds_i = np.mean(psds_reg[area == AOI[i], :], axis=0)
        tau, fok = compute_tau(f_axis, psds_i)
        tau_lst[i] = tau
        fok_lst[i] = fok
    return tau_lst, fok_lst

def plot_tau_feature(tau_lst, feature, areas,
                    x_label='Gradient of Embedding', y_label='Feature',
                    annotate_flag=False, log_flag=False, reverse_xy_flag=False, p_fit=True, loglog_flag=False):
    fig, ax = plt.subplots(figsize=(15,10), facecolor=(1, 1, 1))
    feature = np.array(feature); tau_lst = np.array(tau_lst)
    if reverse_xy_flag:
        tau_lst, feature = feature, tau_lst
        x_label, y_label = y_label, x_label
    scatter = ax.scatter(feature, tau_lst)
    if annotate_flag:
        for i in range(len(areas)):
            plt.annotate(areas[i], (feature[i], tau_lst[i]), rotation=0,
                         horizontalalignment='left', verticalalignment='top',
                         fontsize = 18)
    if log_flag:
        plt.yscale('log')
#         plt.text(0.1, 0.9, "r^2 = {:.2f}".format(np.power(ma.corrcoef(ma.masked_invalid(feature),
#                                                      ma.masked_invalid(np.log10(tau_lst)))[0, 1], 2)), transform=plt.gca().transAxes, fontsize=20)
#     else:
#         plt.text(0.1, 0.9, "r^2 = {:.2f}".format(np.power(ma.corrcoef(ma.masked_invalid(feature),
#                                                      ma.masked_invalid(tau_lst))[0, 1], 2)), transform=plt.gca().transAxes, fontsize=20)
    if p_fit:
        if loglog_flag:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(feature), np.log10(tau_lst))
            plt.plot(feature, np.power(10, slope*np.log10(feature) + intercept), color='red') 
            plt.xscale('log')
        else:
            if log_flag:
                slope, intercept, r_value, p_value, std_err = stats.linregress(feature, np.log10(tau_lst))
                plt.plot(feature, np.power(10, slope*feature + intercept), color='red') 
            else:
                slope, intercept, r_value, p_value, std_err = stats.linregress(feature, tau_lst)
                plt.plot(feature, slope*feature + intercept, color='red') 

        # Print the p-value on the figure
        r2_value = np.power(r_value, 2)
        plt.text(0.1, 0.9, f'r^2-value: {r2_value:.2f}', transform=plt.gca().transAxes, fontsize=20)
        plt.text(0.1, 0.85, f'p-value: {p_value:.2e}', transform=plt.gca().transAxes, fontsize=20)
    # handles, labels_c = scatter.legend_elements()
    # labels = np.array(list(Region_Color_dict.keys()))
    # values = np.array(list(Region_Color_dict.values()))
    # sort_val_ind = np.argsort(values)
    # labels = labels[sort_val_ind]
    # legend = ax.legend(handles, labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make the bottom and left spines thicker
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    return fig

def plot_line_curve(tau_lst, feature, areas, marker = '-',
                    x_label='Gradient of Embedding', y_label='Feature',
                    annotate_flag=False, log_flag=False, reverse_xy_flag=False):
    fig, ax = plt.subplots(figsize=(15,10), facecolor=(1, 1, 1))
    if reverse_xy_flag:
        tau_lst, feature = feature, tau_lst
        x_label, y_label = y_label, x_label
    scatter = ax.plot(feature, tau_lst, marker, ms=20)
    if annotate_flag:
        for i in range(len(areas)):
            plt.annotate(areas[i], (feature[i], tau_lst[i]), rotation=0,
                         horizontalalignment='left', verticalalignment='top',
                         fontsize = 18)
    if log_flag:
        plt.yscale('log')
        plt.title("Corr = {:.2f}".format(ma.corrcoef(ma.masked_invalid(feature),
                                                     ma.masked_invalid(np.log10(tau_lst)))[0, 1]), fontsize=20)
    else:
        plt.title("Corr = {:.2f}".format(ma.corrcoef(ma.masked_invalid(feature),
                                                     ma.masked_invalid(tau_lst))[0, 1]), fontsize=20)
    # handles, labels_c = scatter.legend_elements()
    # labels = np.array(list(Region_Color_dict.keys()))
    # values = np.array(list(Region_Color_dict.values()))
    # sort_val_ind = np.argsort(values)
    # labels = labels[sort_val_ind]
    # legend = ax.legend(handles, labels)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Make the bottom and left spines thicker
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    return fig

def save_fig(name):
    # Save the figure as a high-resolution PNG
    plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    # Save the figure as a high-resolution SVG
    plt.savefig(name+'.svg', dpi=300, bbox_inches='tight')
    # plt.close()
    
    return

def plot_dynamics(p, r_exc, r_inh, I_stim_exc, area_stim_idx, area_name_list, t_plot, PULSE_INPUT):
    area_idx_list=[-1]
    clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
    for name in area_name_list:
        area_idx_list=area_idx_list+[p['areas'].index(name)]
    f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(12, 12), facecolor=(1, 1, 1))

    # clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
    for ax, area_idx in zip(ax_list, area_idx_list):
        if area_idx < 0:
            y_plot = I_stim_exc[:, area_stim_idx].copy()
            z_plot = np.zeros_like(y_plot)
            txt = 'Input'
            color = 'k'
        else:
            y_plot = r_exc[:,area_idx].copy()
            z_plot = r_inh[:,area_idx].copy()
            txt = p['areas'][area_idx]
            color = clist[0][area_idx]

        if PULSE_INPUT:
            y_plot = y_plot - y_plot.min()
            # y_plot = y_plot - 10
            z_plot = z_plot - z_plot.min()
            ax.plot(t_plot, y_plot,color=color, linewidth=1.5)
            #ax.plot(t_plot, z_plot,'--',color='b')
        else:
            #ax.plot(t_plot, y_plot,color='r')
            ax.plot(t_plot, y_plot,color=color, linewidth=1)
            # ax.plot(t_plot[0:10000], y_plot[-1-10000:-1],color=color, linewidth=2)
            # ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')

        # ax.plot(t_plot, y_plot,color=clist[0][c_color])
        # ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
        # c_color=c_color+1
        ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=20)
        # Hide the top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Make the bottom and left spines thicker
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

    f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=20)
    ax.set_xlabel('Time (ms)', fontsize=20) 
    return f



def plot_IPR_theta(ipr_eigenvecs, theta_eigenvecs, eigVals_slow, flag_norm=0, norm_lim=[4,8]):
    fig = plt.subplots(1, 1,figsize=(14, 10), facecolor=(1, 1, 1))
    tau_slow = -1/np.real(eigVals_slow)
    tau_log = np.log(tau_slow)

    if flag_norm == 1:
        norm = plt.Normalize(norm_lim[0], norm_lim[1])
    else:
        norm = plt.Normalize(tau_log.min(), tau_log.max())

    ax = sns.scatterplot(x=np.log(ipr_eigenvecs), y=theta_eigenvecs, hue=np.log(tau_slow),s=400, palette='rainbow', hue_norm=norm)

    sm = plt.cm.ScalarMappable(cmap="rainbow", norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove()
    cbar = ax.figure.colorbar(sm)
    cbar.ax.tick_params(labelsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('log(IPR)', fontsize=20)
    plt.ylabel('theta', fontsize=20)
    plt.xlim([-4, 0.05])
    plt.ylim([0.4, 1.05])

    # # Hide the top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Make the bottom and left spines thicker
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    return fig, ax
