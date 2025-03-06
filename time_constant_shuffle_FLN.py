# -*- coding: utf-8 -*-
"""
From Spyder Editor

Author: Guanchun Li (NYU Courant)

Time: Sat May  8 13:04:04 2021
"""

import pre_functions_clean as pf
import matplotlib.pyplot as plt
import numpy as np
import scipy 
from scipy import stats
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import random

from numpy import linspace
import statsmodels.tsa.api as smt
from scipy import optimize
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_W_EI_shuffle_fln(p_t,FLN_shuffled_t,LONG_RANGE_IDENTICAL_HIERARCHY=0):
	
	p=p_t.copy()
	FLN_shuffled=FLN_shuffled_t.copy()
	
	if LONG_RANGE_IDENTICAL_HIERARCHY:
		lr_id_hierarchy = np.ones(len(p['areas']))*np.mean(p['hier_vals'])
		p['exc_scale'] = (1+p['eta']*lr_id_hierarchy)
		p['inh_scale'] = (1+p['eta_inh']*lr_id_hierarchy)
	else:
		p['exc_scale'] = (1+p['eta']*p['hier_vals'])
		p['inh_scale'] = (1+p['eta_inh']*p['hier_vals_inh'])
	p['local_exc_scale'] = (1+p['eta_local']*p['hier_vals'])
	p['local_inh_scale'] = (1+p['eta_inh_local']*p['hier_vals'])

	local_EE =  p['beta_exc'] * p['wEE'] * p['local_exc_scale']
	local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
	local_IE =  p['beta_inh'] * p['wIE'] * p['local_inh_scale']
	local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)

	fln_scaled = (p['exc_scale'] * FLN_shuffled.T).T
	fln_scaled_inh = (p['inh_scale'] * FLN_shuffled.T).T

	W=np.zeros((2*p['n_area'],2*p['n_area']))		
	
	for i in range(p['n_area']):
		W[2*i,2*i]=(local_EE[i]-1)/p['tau_exc']
		W[2*i,2*i+1]=local_EI[i]/p['tau_exc']
		W[2*i+1,2*i+1]=(local_II[i]-1)/p['tau_inh']
		W[2*i+1,2*i]=local_IE[i]/p['tau_inh']

		W[2*i,::2]=W[2*i,::2]+p['beta_exc']*p['muEE']*fln_scaled[i,:]/p['tau_exc']
		W[2*i+1,::2]=W[2*i+1,::2]+p['beta_inh']*p['muIE']*fln_scaled_inh[i,:]/p['tau_inh']
		
	W_EI=np.zeros_like(W)
	W_EI[0:p['n_area'],0:p['n_area']]=W.copy()[0::2,0::2]
	W_EI[0:p['n_area'],p['n_area']:]=W.copy()[0::2,1::2]
	W_EI[p['n_area']:,0:p['n_area']]=W.copy()[1::2,0::2]
	W_EI[p['n_area']:,p['n_area']:]=W.copy()[1::2,1::2]
	
	return W_EI

def IPR_effective_weight(p_t, FLN_shuffled_t, lambda_bias=0):
	W_EI = generate_W_EI_shuffle_fln(p_t,FLN_shuffled_t)
	W_EE = W_EI[0:p_t['n_area'], 0:p_t['n_area']]
	W_EE_effect = W_EE - lambda_bias * np.eye(p_t['n_area'])
	W_EE_normed = pf.normalize_matrix(W_EE_effect)
	IPR_W_EE = pf.IPR_MATRIX(W_EE_normed)[0]
	return IPR_W_EE

def eigen_structure_approximation(p_t, FLN_t, MACAQUE_CASE=1,SHUFFLE_FLN=0,SHUFFLE_TYPE=0,CONSENSUS_CASE=0):

	p = p_t.copy()
	FLN = FLN_t.copy()
	
	# _,W0=pf.genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=1,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
	# p,W1=pf.genetate_net_connectivity(p,LOCAL_IDENTICAL_HIERARCHY=0,ZERO_FLN=0,SHUFFLE_FLN=SHUFFLE_FLN,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE,STRONG_GBA=0,CONSENSUS_CASE=CONSENSUS_CASE)
	
	theta = p['beta_exc']*p['muEE']/p['tau_exc']/(p['beta_inh']*p['muIE']/p['tau_inh'])
	# print('theta=',theta)
	
	# #---------------------------------------------------------------------------------
	# #reshape the connectivity matrix by E and I population blocks, EE, EI, IE, II
	# #---------------------------------------------------------------------------------
	# W0_EI=np.zeros_like(W0)
	# W0_EI[0:p['n_area'],0:p['n_area']]=W0.copy()[0::2,0::2]
	# W0_EI[0:p['n_area'],p['n_area']:]=W0.copy()[0::2,1::2]
	# W0_EI[p['n_area']:,0:p['n_area']]=W0.copy()[1::2,0::2]
	# W0_EI[p['n_area']:,p['n_area']:]=W0.copy()[1::2,1::2]
	
	# W1_EI=np.zeros_like(W1)
	# W1_EI[0:p['n_area'],0:p['n_area']]=W1.copy()[0::2,0::2]
	# W1_EI[0:p['n_area'],p['n_area']:]=W1.copy()[0::2,1::2]
	# W1_EI[p['n_area']:,0:p['n_area']]=W1.copy()[1::2,0::2]
	# W1_EI[p['n_area']:,p['n_area']:]=W1.copy()[1::2,1::2]
	
	W0_EI = generate_W_EI_shuffle_fln(p, 0*FLN)
	W1_EI = generate_W_EI_shuffle_fln(p, FLN)
	
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

	#--------------------------------------------------------------------------
	#extract block matrices after similarity transformation on the connectivity matrix
	#--------------------------------------------------------------------------
	Lambda_1 = D_EE + A@D_IE
	Sigma_1 = (F_EE + A@F_IE)@(np.eye(p['n_area'])+A@B)
	W_app = np.matmul(np.eye(p['n_area'])+A@B, np.matmul(Lambda_1 + Sigma_1,
					  np.linalg.inv(np.eye(p['n_area'])+A@B)))
	
	lambda_lst, eigVecs = np.linalg.eig(Lambda_1 + Sigma_1)
	lambda_app_lst = np.zeros([p['n_area'], ])
	for k in range(p['n_area']):
		lambda_k = Lambda_1[k, k]
		for j in range(p['n_area']):
			if j != k:
				ind_k =	 max(np.abs(Sigma_1[k, j]), np.abs(Sigma_1[j, k])) < 3*np.abs(Lambda_1[k, k] - Lambda_1[j, j])
				ind_k = 1
				lambda_k += ind_k * Sigma_1[k, j] * Sigma_1[j, k] / (Lambda_1[k, k] - Lambda_1[j, j])
		lambda_app_lst[k] = lambda_k
	return lambda_app_lst, lambda_lst, Sigma_1, W_app
	
def time_constant_shuffle_fln(p_t,FLN_shuffled_t,theta_FLAG=0, eigVecs_Flag=0,abs_Flag=1):
	
	W_EI = generate_W_EI_shuffle_fln(p_t,FLN_shuffled_t)
	p=p_t.copy()

	#---------------------------------------------------------------------------------
	# eigenmode decomposition
	#--------------------------------------------------------------------------------- 
	eigVals, eigVecs = np.linalg.eig(W_EI)
	eigVecs_inv = np.linalg.inv(eigVecs)
	if abs_Flag:	
		eigVecs_a=np.abs(eigVecs)
		eigVecs_inv_a=np.abs(eigVecs_inv)
	else:
		eigVecs_a=eigVecs
		eigVecs_inv_a=eigVecs_inv
		
	tau=-1/np.real(eigVals)
	tau_s=np.zeros_like(tau)
	for i in range(len(tau)):
		tau_s[i]=format(tau[i],'.2f')

	ind=np.argsort(-tau_s)
	eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVecs_inv_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVals_reorder=np.zeros((2*p['n_area'],))+0j
	
	for i in range(2*p['n_area']):
		eigVals_reorder[i] = eigVals[ind[i]]
		eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
		eigVecs_inv_a_reorder[i, :]=eigVecs_inv_a[ind[i], :]
		
	eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
	eigVecs_inv_slow=eigVecs_inv_a_reorder[:p['n_area'],:p['n_area']]
	eigVals_slow=eigVals_reorder[:p['n_area']]
	
	eigVecs_slow_normed=pf.normalize_matrix(eigVecs_slow,column=1) #normalized by column
	eigVecs_inv_slow_normed=pf.normalize_matrix(eigVecs_inv_slow,column=1) #normalized by column
	
	eigVecs_slow_normed=np.fliplr(eigVecs_slow_normed)	#flip the IPRs of each area starts from V1, but no need to flip green because green func is sorted by row, which naturally starts from V1
	eigVecs_inv_slow_normed=np.flipud(eigVecs_inv_slow_normed)
	eigVals_slow = np.flip(eigVals_slow)
	
	ipr_eigenvecs=np.zeros(p['n_area'])
	theta_eigenvecs=np.zeros(p['n_area'])
	#normalize the coefficient row by row
	for j in range(p['n_area']):
		ipr_eigenvecs[j]=pf.IPR(np.abs(eigVecs_slow_normed[:,j])) #normailzied by column
		# ipr_eigenvecs[j]=IPR_est(np.abs(eigVecs_slow_normed[:,j])) #normailzied by column
		if theta_FLAG:
			theta_eigenvecs[j]=pf.THETA(eigVecs_slow_normed[:,j], p['full_dist_mat'])
	if theta_FLAG:
		if eigVecs_Flag:
			return ipr_eigenvecs, eigVals_slow, theta_eigenvecs, eigVecs_slow_normed, eigVecs_inv_slow_normed
		else:
			return ipr_eigenvecs, eigVals_slow, theta_eigenvecs
	else:
		if eigVecs_Flag:
			return ipr_eigenvecs, eigVals_slow, eigVecs_slow_normed, eigVecs_inv_slow_normed
		else:
			return ipr_eigenvecs, eigVals_slow


def time_constant_module_shuffle_fln(p_t,FLN_shuffled_t,tau_flag=False,theta_flag=False):
	
	W_EI = generate_W_EI_shuffle_fln(p_t,FLN_shuffled_t)
	p=p_t.copy()

	#---------------------------------------------------------------------------------
	# eigenmode decomposition
	#--------------------------------------------------------------------------------- 
	eigVals, eigVecs = np.linalg.eig(W_EI)
	eigVecs_inv = np.linalg.inv(eigVecs)

	eigVecs_a=eigVecs
	eigVecs_inv_a=eigVecs_inv
		
	tau=-1/np.real(eigVals)
	tau_s=np.zeros_like(tau)
	for i in range(len(tau)):
		tau_s[i]=format(tau[i],'.2f')
		
	if np.min(tau) < 0:
		print("Unstable!")

	ind=np.argsort(-tau_s)
	eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVecs_inv_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVals_reorder=np.zeros((2*p['n_area'],))+0j
	
	for i in range(2*p['n_area']):
		eigVals_reorder[i] = eigVals[ind[i]]
		eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
		eigVecs_inv_a_reorder[i, :]=eigVecs_inv_a[ind[i], :]
		
	eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
	eigVecs_inv_slow=eigVecs_inv_a_reorder[:p['n_area'],:p['n_area']]
	eigVals_slow=eigVals_reorder[:p['n_area']]
	
	eigVecs_slow_normed=pf.normalize_matrix(eigVecs_slow,column=1) #normalized by column
	eigVecs_inv_slow_normed=pf.normalize_matrix(eigVecs_inv_slow,column=1) #normalized by column
	
	eigVecs_slow_normed=np.fliplr(eigVecs_slow_normed)	#flip the IPRs of each area starts from V1, but no need to flip green because green func is sorted by row, which naturally starts from V1
	eigVecs_inv_slow_normed=np.flipud(eigVecs_inv_slow_normed)
	eigVals_slow = np.flip(eigVals_slow)
	
	arg_max = np.argmax(np.abs(eigVecs_slow_normed), 0)
	arg_sign = np.sign(np.real(eigVecs_slow_normed)[arg_max, np.arange(p['n_area'])])
	eigVecs_signed = eigVecs_slow_normed * arg_sign
	
	ipr_eigenvecs=np.zeros(p['n_area'])
	ipr_pos_eigenvecs=np.zeros(p['n_area'])
	ipr_neg_eigenvecs=np.zeros(p['n_area'])
	ipr_weight=np.zeros(p['n_area'])
	theta_eigenvecs=np.zeros(p['n_area'])
	#theta_eigenvecs=np.zeros(p['n_area'])
	#normalize the coefficient row by row
	for j in range(p['n_area']):
		eigVecs_j = eigVecs_signed[:,j]
		eigVecs_j_pos = eigVecs_j[np.real(eigVecs_j) > 0]
		ipr_eigenvecs[j]=pf.IPR(np.abs(eigVecs_j)) #normailzied by column
		ipr_pos_eigenvecs[j]=pf.IPR(np.abs(eigVecs_j[np.real(eigVecs_j) > 0]))
		ipr_neg_eigenvecs[j]=pf.IPR(np.abs(eigVecs_j[np.real(eigVecs_j) < 0]))
		ipr_weight[j]=(np.linalg.norm(eigVecs_j[np.real(eigVecs_j) < 0])/np.linalg.norm(eigVecs_j[np.real(eigVecs_j) > 0]))**2
		if theta_flag:
			theta_eigenvecs[j]=pf.THETA(np.abs(eigVecs_j), p['full_dist_mat'])
		# ipr_eigenvecs[j]=IPR_est(np.abs(eigVecs_slow_normed[:,j])) #normailzied by column
	if tau_flag:
		if theta_flag:
			return ipr_eigenvecs, ipr_pos_eigenvecs, ipr_neg_eigenvecs, ipr_weight, eigVals_slow, eigVecs_slow_normed, tau_s, theta_eigenvecs
		else:
			return ipr_eigenvecs, ipr_pos_eigenvecs, ipr_neg_eigenvecs, ipr_weight, eigVals_slow, eigVecs_slow_normed, tau_s
	else:
		if theta_flag:
			return ipr_eigenvecs, ipr_pos_eigenvecs, ipr_neg_eigenvecs, ipr_weight, eigVals_slow, eigVecs_slow_normed, theta_eigenvecs
		else:
			return ipr_eigenvecs, ipr_pos_eigenvecs, ipr_neg_eigenvecs, ipr_weight, eigVals_slow, eigVecs_slow_normed



def time_constant_area_shuffle_fln(p_t,FLN_shuffled_t):
	
	W_EI = generate_W_EI_shuffle_fln(p_t,FLN_shuffled_t)
	p=p_t.copy()

	#---------------------------------------------------------------------------------
	# eigenmode decomposition
	#--------------------------------------------------------------------------------- 
	eigVals, eigVecs = np.linalg.eig(W_EI)
	eigVecs_inv = np.linalg.inv(eigVecs)

	eigVecs_a=eigVecs
	eigVecs_inv_a=eigVecs_inv

	tau=-1/np.real(eigVals)
	tau_s=np.zeros_like(tau)
	for i in range(len(tau)):
		tau_s[i]=format(tau[i],'.2f')


	ind=np.argsort(-tau_s)
	eigVecs_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVecs_inv_a_reorder=np.zeros((2*p['n_area'],2*p['n_area']))+0j
	eigVals_reorder=np.zeros((2*p['n_area'],))+0j
	
	for i in range(2*p['n_area']):
		eigVals_reorder[i] = eigVals[ind[i]]
		eigVecs_a_reorder[:,i]=eigVecs_a[:,ind[i]]
		eigVecs_inv_a_reorder[i, :]=eigVecs_inv_a[ind[i], :]
		
	eigVecs_slow=eigVecs_a_reorder[:p['n_area'],:p['n_area']]
	eigVecs_inv_slow=eigVecs_inv_a_reorder[:p['n_area'],:p['n_area']]
	eigVals_slow=eigVals_reorder[:p['n_area']]
	
	# eigVecs_slow_normed=pf.normalize_matrix(eigVecs_slow,column=1) #normalized by column
	# eigVecs_inv_slow_normed=pf.normalize_matrix(eigVecs_inv_slow,column=1) #normalized by column
	
	eigVecs_slow=np.fliplr(eigVecs_slow)  #flip the IPRs of each area starts from V1, but no need to flip green because green func is sorted by row, which naturally starts from V1
	eigVecs_inv_slow=np.flipud(eigVecs_inv_slow)
	eigVals_slow = np.flip(eigVals_slow)
	eigVals_slow_real = np.real(eigVals_slow)
	
	theta_area_eigenvecs=np.zeros(p['n_area'])
	Green_coef=np.zeros([p['n_area'], p['n_area']])
	#theta_eigenvecs=np.zeros(p['n_area'])
	#normalize the coefficient row by row
	for j in range(p['n_area']):
		Green_coef[j,:] = np.real(eigVecs_slow[j, :] * eigVecs_inv_slow[:, j])
		# Green_coef[j,:] = np.real(eigVecs_slow[j, :] * eigVecs_inv_slow[:, 0])  # visual stimulus
		# Green_coef[j,:] = np.real(eigVecs_slow[j, :] * eigVecs_inv_slow[:, 9])  # macaque sensory stimulus
		# Green_coef[j,:] = np.real(eigVecs_slow[j, :] * eigVecs_inv_slow[:, 4])  # marmoset sensory stimulus
		theta_area_eigenvecs[j] = theta_area(Green_coef[j,:], eigVals_slow_real)
		# ipr_eigenvecs[j]=IPR_est(np.abs(eigVecs_slow_normed[:,j])) #normailzied by column

	return theta_area_eigenvecs, Green_coef, eigVals_slow_real

def theoretical_time_constant_input_at_all_areas(p_t,eigVecs,eigVals, T_lag=int(1e3), plot_Flag=0):
	p=p_t.copy()
	area_name_list=p['areas']
	
	print('inv_cond=',np.linalg.cond(eigVecs))
	inv_eigVecs=np.linalg.inv(eigVecs)
	
	n=len(area_name_list)
	# T_lag=int(1e3)
			
	acf_data=np.zeros((n,T_lag+1))+0j 
	coef=np.zeros((n,2*n))+0j
	coef_green=np.zeros((n,2*n))+0j	 #cofficient of the green's function 
	#m=0  #9 for area 2
	for i in np.arange(n):
		m=i
		for j in np.arange(2*n):
			coef_green[i,j]=eigVecs[i,j]*inv_eigVecs[j,m]
			coef[i,j]=0
			for k in np.arange(2*n):
				coef[i,j] += eigVecs[i,j]*eigVecs[i,k]*inv_eigVecs[j,m]*inv_eigVecs[k,m]/(-eigVals[j]-eigVals[k])
			for s in np.arange(T_lag+1):
				acf_data[i,s] += coef[i,j]*np.exp(eigVals[j]*s)
		acf_data[i,:]=acf_data[i,:]/acf_data[i,0]
	
	if plot_Flag == 1:	 
		clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_name_list)))[np.newaxis, :, :3]
		
		plt.figure(figsize=(10,5))
		ax = plt.axes()
		for i in np.arange(len(area_name_list)):
			plt.plot(np.arange(T_lag+1),np.real(acf_data[i,:]),color=clist[0][i])
			
		# plt.legend(area_name_list)
		plt.xlabel('Time difference (ms)')
		plt.ylabel(' Theoretical correlation')
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		plt.title('input loc and corr measure at the same area')
		#plt.savefig('result/correlation_stim_V1.pdf')
#	
#	print('Start exp fitting!')
#	
	print('Start exp fitting!'); t_plot=np.arange(T_lag)
#	 
	delay_time=np.zeros(len(area_name_list)); acf_data = np.real(acf_data)
	# f, ax_list = plt.subplots(len(area_name_list), sharex=True, figsize=(15,15))
	for i in np.arange(len(area_name_list)):
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
				
		# print('error ratio of',area_name_list[i],"=",str(e_ratio))
#		
#		ax.plot(t_plot[0:p_end],acf_data[i,0:p_end])
#		ax.plot(t_plot[0:p_end],r_single[1]*np.exp(-t_plot[0:p_end]/r_single[0]),'r--')
#		ax.plot(t_plot[0:p_end],r_double[2]*np.exp(-t_plot[0:p_end]/r_double[0])-r_double[3]*np.exp(-t_plot[0:p_end]/r_double[1]),'g--')
#		ax.set_ylim(0,1)
#		txt = area_name_list[i]
#		ax.text(0.9, 0.6, txt, transform=ax.transAxes)
#		
#	f.text(0.01, 0.5, 'Theoretical Correlation', va='center', rotation='vertical')
#	ax.set_xlabel('Time difference (ms)')
#	ax.set_title('input_loc and corr measure at the same area')
#	
#	plt.figure(figsize=(5,7))		 
#	ax = plt.axes()
#	plt.bar(np.arange(len(area_name_list)),delay_time,width = 1,color=clist[0])
#	plt.xticks(np.arange(len(area_name_list)),area_name_list,rotation=90)
#	plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
#	plt.ylabel('Theoretical $T_{delay}$ (ms)')
#	ax.spines['top'].set_visible(False)
#	ax.spines['right'].set_visible(False)
#	ax.set_title('input_loc and corr measure at the same area')
#	
#	ind=np.argsort(delay_time)
#	delay_time_sort=np.zeros_like(delay_time)
#	area_name_list_sort=[]
#	for i in np.arange(len(ind)):
#		delay_time_sort[i]=delay_time[ind[i]]
#		area_name_list_sort=area_name_list_sort+[area_name_list[ind[i]]]
#		
#	plt.figure(figsize=(5,7))		 
#	ax = plt.axes()
#	plt.bar(np.arange(len(area_name_list)),delay_time_sort,width = 1,color=clist[0])
#	plt.xticks(np.arange(len(area_name_list)),area_name_list_sort,rotation=90)
#	plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
#	plt.ylabel('Theoretical $T_{delay}$ (ms)')
#	ax.spines['top'].set_visible(False)
#	ax.spines['right'].set_visible(False)
#	ax.set_title('input_loc and corr measure at the same area')
	
	tau_s=np.zeros(2*n)
	for i in range(len(tau_s)):
		tau_s[i]=float(format(np.real(-1/eigVals[i]),'.2f'))
	eigVals_slow = np.sort(np.real(eigVals))[p['n_area']:]; tau_s = np.real(-1/eigVals_slow)
	# ind=np.argsort(-tau_s)
	# coef_reorder=np.zeros((p['n_area'],2*p['n_area']))+0j
	# coef_green_reorder=np.zeros((p['n_area'],2*p['n_area']))+0j
	# tau_reorder=np.zeros(2*p['n_area'])
	
	# for i in range(2*p['n_area']):
	#	  coef_reorder[:,i]=coef[:,ind[i]]
	#	  coef_green_reorder[:,i]=coef_green[:,ind[i]]
	#	  tau_reorder[i]=tau_s[ind[i]]
	
	# coef_normed=np.zeros_like(coef)
	# coef_normed_green=np.zeros_like(coef_green)

	# #normalize the coefficient row by row
	# for j in range(p['n_area']):
	#	  coef_normed[j,:]=coef_reorder[j,:]/np.max(np.abs(coef_reorder[j,:]))
	#	  coef_normed_green[j,:]=coef_green_reorder[j,:]/np.max(np.abs(coef_green_reorder[j,:]))
		
	# fig, ax = plt.subplots(figsize=(20,10))
	# f=ax.pcolormesh(np.abs(coef_normed),cmap='hot')
	# fig.colorbar(f,ax=ax,pad=0.1)
	# ax.set_title('full coef matrix of autocorrelation')
		
	# x = np.arange(2*p['n_area']) # xticks
	# y = np.arange(p['n_area']) # yticks
	# xlim = (0,2*p['n_area'])
	# ylim = (0,p['n_area'])
	
	# yticklabels_odd=p['areas'][1::2]
	# yticklabels_even=p['areas'][::2]
	
	# # set original ticks and ticklabels
	# ax.set_xlim(xlim)
	# ax.set_ylim(ylim)
	# ax.set_xticks(x[::1])
	# ax.invert_xaxis()
		
	# ax.set_xticklabels(tau_reorder)
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
	
		
	# fig, ax = plt.subplots(figsize=(12,10))
	# f=ax.pcolormesh(np.abs(coef_normed[:,:p['n_area']]),cmap='hot')
	# fig.colorbar(f,ax=ax,pad=0.1)
	# ax.set_title('E population coef matrix of autocorrelation')
		
	# x = np.arange(p['n_area']) # xticks
	# y = np.arange(p['n_area']) # yticks
	# xlim = (0,p['n_area'])
	# ylim = (0,p['n_area'])
	
	# yticklabels_odd=p['areas'][1::2]
	# yticklabels_even=p['areas'][::2]
	
	# # set original ticks and ticklabels
	# ax.set_xlim(xlim)
	# ax.set_ylim(ylim)
	# ax.set_xticks(x[::1])
	# ax.invert_xaxis()
		
	# ax.set_xticklabels(tau_reorder[:p['n_area']])
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
	
	
	# fig, ax = plt.subplots(figsize=(12,10))
	# f=ax.pcolormesh(np.abs(coef_normed_green[:,:p['n_area']]),cmap='hot')
	# fig.colorbar(f,ax=ax,pad=0.1)
	# ax.set_title('E population coef matrix of Greens func')
		
	# x = np.arange(p['n_area']) # xticks
	# y = np.arange(p['n_area']) # yticks
	# xlim = (0,p['n_area'])
	# ylim = (0,p['n_area'])
	
	# yticklabels_odd=p['areas'][1::2]
	# yticklabels_even=p['areas'][::2]
	
	# # set original ticks and ticklabels
	# ax.set_xlim(xlim)
	# ax.set_ylim(ylim)
	# ax.set_xticks(x[::1])
	# ax.invert_xaxis()
		
	# ax.set_xticklabels(tau_reorder[:p['n_area']])
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
	return coef, delay_time, acf_data, tau_s
	
def role_of_connection_by_shuffling_FLN(p_t,plot_Flag=1, MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0, n_trial=1000, theta_FLAG=0):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	
	if MACAQUE_CASE:
		hi_threshold = 0.
	else:
		hi_threshold = 0.
	# tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)
	if theta_FLAG:
		[ipr_eig_ori,eigVals_ori,theta_ori]=time_constant_shuffle_fln(p,fln_mat,theta_FLAG=1)
	else:
		[ipr_eig_ori,eigVals_ori]=time_constant_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)
	
	# Compute the material cost
	material_cost_ori = np.sum(p['fln_mat']*p['full_dist_mat'][:p['n_area'],:p['n_area']])
	# Compute the time constant range 
	tau_ori = -1 / np.real(eigVals_ori)
	tau_std_ori = np.std(tau_ori)

	# n_trial=1000
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))

	
	mean_ipr_eig=np.zeros(n_trial)
	mean_ipr_eig_r=np.zeros(n_trial)
	mean_ipr_eig_i=np.zeros(n_trial)
	num_imag_eig=np.zeros(n_trial)
	sym_idx_lst=np.zeros(n_trial)
	mean_theta_eig=np.zeros(n_trial)
	ipr_hi_corr=np.zeros(n_trial)
	
	material_cost=np.zeros(n_trial)
	tau_std=np.zeros(n_trial)

	eigVals_complex=np.array([])
	eigVals_total=np.array([])
	
	for j in np.arange(n_trial):   
		print('n_trial=',j)
		
		max_eigval=1
		while max_eigval>-5e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=pf.matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
			# max_eigval = - 1e-4
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			# print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		sym_idx_lst[j]=matrix_symmetric_index(fln_shuffled)
		p['fln_mat']=fln_shuffled
		if theta_FLAG:
			ipr_eig_shuffled_j, eigVals_shuffled_j, theta_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled,theta_FLAG=theta_FLAG)
			mean_theta_eig[j] = np.mean(theta_shuffled_j)
		else:
			ipr_eig_shuffled_j, eigVals_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled)

		# Compute the material cost
		material_cost[j] = np.sum(p['fln_mat']*p['full_dist_mat'][:p['n_area'],:p['n_area']])
		# Compute the time constant range 
		tau_shuffled_j = -1 / np.real(eigVals_shuffled_j)
		tau_std[j] = np.std(tau_shuffled_j)

		# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
		eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
		# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
		ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
		eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
		eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
		# ind_pickup_j = [True]*p['n_area']
		mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
		mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
		mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
		num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
		hi_index = p['hier_vals'] > hi_threshold
		ipr_hi_corr[j]=np.corrcoef([p['hier_vals'][hi_index], ipr_eig_shuffled_j[hi_index]])[0, 1]
	
	mean_ipr_total = mean_ipr_eig
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	if plot_Flag:
		# fig,ax=plt.subplots(1,4,figsize=(40,8)) 
		# ax[0].hist(mean_ipr_eig,15,rwidth=0.8,facecolor='r',alpha=0.5)
		# ax[0].set_xlabel('mean ipr', fontsize=18)
		# ax[0].set_ylabel('Count', fontsize=18)
		# ax[0].tick_params(axis='x', labelsize=16)
		# ax[0].tick_params(axis='y', labelsize=16)
		# #ax[0].set_xlim([0,1])
		# ax[0].axvline(np.mean(ipr_eig_ori))
		
		# ax[1].hist(material_cost,15,rwidth=0.8,facecolor='g',alpha=0.5)
		# ax[1].set_xlabel('Material Cost', fontsize=18)
		# ax[1].set_ylabel('Count', fontsize=18)
		# ax[1].tick_params(axis='x', labelsize=16)
		# ax[1].tick_params(axis='y', labelsize=16)
		# #ax[0].set_xlim([0,1])
		# ax[1].axvline(material_cost_ori)

		# ax[2].hist(tau_std,15,rwidth=0.8,facecolor='b',alpha=0.5)
		# ax[2].set_xlabel('tau standard deviation', fontsize=18)
		# ax[2].set_ylabel('Count', fontsize=18)
		# ax[2].tick_params(axis='x', labelsize=16)
		# ax[2].tick_params(axis='y', labelsize=16)
		# #ax[0].set_xlim([0,1])
		# ax[2].axvline(tau_std_ori)

		# if theta_FLAG:
		# 	ax[3].hist(mean_theta_eig,15,rwidth=0.8,facecolor='c',alpha=0.5)
		# 	ax[3].set_xlabel('mean theta', fontsize=18)
		# 	ax[3].set_ylabel('Count', fontsize=18)
		# 	ax[3].tick_params(axis='x', labelsize=16)
		# 	ax[3].tick_params(axis='y', labelsize=16)
		# 	ax[3].axvline(np.mean(theta_ori))
				# fig,ax=plt.subplots(1,4,figsize=(40,8))
		fig,ax=plt.subplots(3, 1,figsize=(20, 18))
		ax[0].hist(tau_std,15,rwidth=0.8,facecolor='b',alpha=0.5)
		ax[0].set_xlabel('tau standard deviation', fontsize=20)
		ax[0].set_ylabel('Count', fontsize=20)
		ax[0].tick_params(axis='x', labelsize=20)
		ax[0].tick_params(axis='y', labelsize=20)
		#ax[0].set_xlim([0,1])
		ax[0].axvline(tau_std_ori)

		if theta_FLAG:
			ax[1].hist(mean_theta_eig,15,rwidth=0.8,facecolor='c',alpha=0.5)
			ax[1].set_xlabel('mean theta', fontsize=20)
			ax[1].set_ylabel('Count', fontsize=20)
			ax[1].tick_params(axis='x', labelsize=20)
			ax[1].tick_params(axis='y', labelsize=20)
			ax[1].axvline(np.mean(theta_ori))
		
		ax[2].hist(mean_ipr_eig,15,rwidth=0.8,facecolor='r',alpha=0.5)
		ax[2].set_xlabel('Mean ipr', fontsize=20)
		ax[2].set_ylabel('Count', fontsize=20)
		ax[2].tick_params(axis='x', labelsize=20)
		ax[2].tick_params(axis='y', labelsize=20)
		#ax[0].set_xlim([0,1])
		ax[2].axvline(np.mean(ipr_eig_ori))

		# ax[1].hist(mean_ipr_eig_i,15,facecolor='r',alpha=0.5)
		# ax[1].set_xlabel('mean ipr eig (complex)')
		# ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
		
		# ax[2].hist(mean_ipr_eig_r,15,facecolor='r',alpha=0.5)
		# ax[2].set_xlabel('mean ipr eig (real)')
		# ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
		
		# ax[3].hist(num_imag_eig,15,facecolor='r',alpha=0.5)
		# ax[3].set_xlabel('number of complex eig')
		# ax[3].axvline(num_imag_eig_ori)
		
		# # ax[2].hist(mean_ipr_eig/np.mean(ipr_eig_ori),15,facecolor='r',alpha=0.5)
		# ax[1].hist(mean_ipr_eig/np.mean(ipr_eig_ori[ind_pickup_ori]),15,facecolor='r',alpha=0.5)
		# ax[1].set_xlabel('normalized mean ipr eig')
		# #ax[2].set_xlim([0,1])
		# ax[1].axvline(1)
		
		#---------------------------------------------------------------------------------
		# plot IPR for shuffled FLNs-II
		#---------------------------------------------------------------------------------
		# mean_ipr_eig=np.zeros(p['n_area']+1)
		# # mean_ipr_green=np.zeros(p['n_area']+1)
	
		# top_90=np.zeros(p['n_area']+1)
		# bottom_10=np.zeros(p['n_area']+1)
		# top_95=np.zeros(p['n_area']+1)
		# bottom_5=np.zeros(p['n_area']+1)
		 
		# plt.figure(figsize=(5,5))		 
		# ax = plt.axes()
		# for k in np.arange(p['n_area']):
		# 	mean_ipr_eig[k]=np.mean(ipr_eig_shuffled[k,:])
		# 	sort_ipr_eig=np.sort(ipr_eig_shuffled[k,:])
		# 	top_90[k]=sort_ipr_eig[int(0.9*n_trial)]
		# 	bottom_10[k]=sort_ipr_eig[int(0.1*n_trial)]
		# 	top_95[k]=sort_ipr_eig[int(0.95*n_trial)]
		# 	bottom_5[k]=sort_ipr_eig[int(0.05*n_trial)]
			
		# 	plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
		# 	plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
		
		# area_avg_ipr_eig=np.mean(ipr_eig_shuffled,0)
		# mean_ipr_eig[p['n_area']]=np.mean(area_avg_ipr_eig)
		# sort_area_avg_ipr_eig=np.sort(area_avg_ipr_eig)
		# top_90[p['n_area']]=sort_area_avg_ipr_eig[int(0.9*n_trial)]
		# bottom_10[p['n_area']]=sort_area_avg_ipr_eig[int(0.1*n_trial)]
		# top_95[p['n_area']]=sort_area_avg_ipr_eig[int(0.95*n_trial)]
		# bottom_5[p['n_area']]=sort_area_avg_ipr_eig[int(0.05*n_trial)]
		
		# plt.vlines(p['n_area'], bottom_10[p['n_area']], top_90[p['n_area']],color="blue",alpha=0.3)
		# plt.vlines(p['n_area'], bottom_5[p['n_area']], top_95[p['n_area']],color="blue",alpha=0.1)
			
		# plt.plot(np.arange(p['n_area']+1),mean_ipr_eig,'.b',markersize=10)	  
		# plt.plot(np.arange(p['n_area']+1),top_90,'.b',markersize=8,alpha=0.3)  
		# plt.plot(np.arange(p['n_area']+1),bottom_10,'.b',markersize=8,alpha=0.3)  
		# plt.plot(np.arange(p['n_area']+1),top_95,'.b',markersize=6,alpha=0.1)  
		# plt.plot(np.arange(p['n_area']+1),bottom_5,'.b',markersize=6,alpha=0.1)	 
		# ipr_eig_ori=np.append(ipr_eig_ori,np.mean(ipr_eig_ori))
		# plt.plot(np.arange(p['n_area']+1),ipr_eig_ori,'.k',markersize=10)
		
		# p['areas'].append('all')
		# # plt.xticks(np.arange(p['n_area']+1),p['areas'],rotation=90)
		# #plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
		# plt.ylabel('eigvector IPR')
		# ax.spines['top'].set_visible(False)
		# ax.spines['right'].set_visible(False)
		
		
		# # #---------------------------------------------------------------------------------
		# # # plot the distribution of complex eigenvalues for shuffled FLNs
		# # #---------------------------------------------------------------------------------
		# plt.figure(figsize=(5,5)) 
		# plt.scatter(np.real(eigVals_complex), np.imag(eigVals_complex))
		# plt.scatter(np.real(eigVals_complex_ori), np.imag(eigVals_complex_ori))
		# plt.title('Distribution of complex eigenvalues')
		# plt.xlabel('Real')
		# plt.ylabel('Imag')
		
		# plt.figure(figsize=(5,5)) 
		# plt.scatter(np.real(eigVals_total), np.imag(eigVals_total))
		# plt.scatter(np.real(eigVals_ori), np.imag(eigVals_ori))
		# plt.title('Distribution of complex eigenvalues')
		# plt.xlabel('Real')
		# plt.ylabel('Imag')
	
	# #---------------------------------------------------------------------------------
	# # plot time constants for shuffled FLNs
	# #---------------------------------------------------------------------------------
	# mean_tau=np.zeros(p['n_area'])
	# top_90=np.zeros(p['n_area'])
	# bottom_10=np.zeros(p['n_area'])
	# top_95=np.zeros(p['n_area'])
	# bottom_5=np.zeros(p['n_area'])
	 
	# plt.figure(figsize=(5,5))		   
	# ax = plt.axes()
	# for k in np.arange(p['n_area']):
	#	  mean_tau[k]=np.mean(tau_shuffled[k,:])
	#	  sort_tau=np.sort(tau_shuffled[k,:])
	#	  top_90[k]=sort_tau[int(0.9*n_trial)]
	#	  bottom_10[k]=sort_tau[int(0.1*n_trial)]
	#	  top_95[k]=sort_tau[int(0.95*n_trial)]
	#	  bottom_5[k]=sort_tau[int(0.05*n_trial)]
		
	#	  plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
	#	  plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
		
		
	# plt.plot(np.arange(p['n_area']),mean_tau,'.b',markersize=10)	  
	# plt.plot(np.arange(p['n_area']),top_90,'.b',markersize=8,alpha=0.3)  
	# plt.plot(np.arange(p['n_area']),bottom_10,'.b',markersize=8,alpha=0.3)  
	# plt.plot(np.arange(p['n_area']),top_95,'.b',markersize=6,alpha=0.1)  
	# plt.plot(np.arange(p['n_area']),bottom_5,'.b',markersize=6,alpha=0.1)	 
	# plt.plot(np.arange(p['n_area']),tau_ori,'.k',markersize=10)
	
	# plt.xticks(np.arange(p['n_area']),p['areas'],rotation=90)
	# plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
	# plt.ylabel('$T_{delay}$ (ms)')
	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	
	return record_fln_shuffled, sym_idx_lst, mean_ipr_total, num_imag_eig, ipr_eig_shuffled, ipr_hi_corr


#visualize the graph of areas and connections for macaque network

def role_of_module_by_shuffling_FLN(p_t,plot_Flag=1, MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0, n_trial=1000):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	
	if MACAQUE_CASE:
		hi_threshold = 0.
	else:
		hi_threshold = 0.
	# tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)
	[ipr_eig_ori, ipr_eig_pos_ori, ipr_eig_neg_ori, ipr_weight_ori, eigVals_ori, eigVecs_ori]=time_constant_module_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)

	# n_trial=1000
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	ipr_eig_pos_shuffled=np.zeros((p['n_area'],n_trial))
	ipr_eig_neg_shuffled=np.zeros((p['n_area'],n_trial))
	ipr_weight_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))

	
	mean_ipr_eig=np.zeros(n_trial)
	mean_ipr_eig_r=np.zeros(n_trial)
	mean_ipr_eig_i=np.zeros(n_trial)
	num_imag_eig=np.zeros(n_trial)
	sym_idx_lst=np.zeros(n_trial)
	mean_theta_eig=np.zeros(n_trial)
	ipr_hi_corr=np.zeros(n_trial)
	
	eigVals_complex=np.array([])
	eigVals_total=np.array([])
	
	for j in np.arange(n_trial):   
		print('n_trial=',j)
		
		max_eigval=1
		while max_eigval>-1e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=pf.matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			# print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		sym_idx_lst[j]=matrix_symmetric_index(fln_shuffled)
		p['fln_mat']=fln_shuffled
		ipr_eig_shuffled_j, ipr_eig_pos_shuffled_j, ipr_eig_neg_shuffled_j, \
		ipr_weight_shuffled_j, eigVals_shuffled_j, eigVecs_shuffled_j=time_constant_module_shuffle_fln(p,fln_shuffled)
		# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
		ipr_eig_pos_shuffled[:, j], ipr_eig_neg_shuffled[:, j] = ipr_eig_pos_shuffled_j, ipr_eig_neg_shuffled_j
		ipr_weight_shuffled[:, j] = ipr_weight_shuffled_j
		eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
		# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
		ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
		eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
		eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
		# ind_pickup_j = [True]*p['n_area']
		mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
		mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
		mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
		num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
		hi_index = p['hier_vals'] > hi_threshold
		ipr_hi_corr[j]=np.corrcoef([p['hier_vals'][hi_index], ipr_eig_shuffled_j[hi_index]])[0, 1]
	
	mean_ipr_total = mean_ipr_eig
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	if plot_Flag:
		fig,ax=plt.subplots(1,4,figsize=(40,8)) 
		ax[0].hist(mean_ipr_eig,15,facecolor='r',alpha=0.5)
		ax[0].set_xlabel('mean ipr eig (total)')
		#ax[0].set_xlim([0,1])
		ax[0].axvline(np.mean(ipr_eig_ori))
		
		
		ax[1].hist(mean_ipr_eig_i,15,facecolor='r',alpha=0.5)
		ax[1].set_xlabel('mean ipr eig (complex)')
		ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
		
		ax[2].hist(mean_ipr_eig_r,15,facecolor='r',alpha=0.5)
		ax[2].set_xlabel('mean ipr eig (real)')
		ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
		
		ax[3].hist(num_imag_eig,15,facecolor='r',alpha=0.5)
		ax[3].set_xlabel('number of complex eig')
		ax[3].axvline(num_imag_eig_ori)
		
		# # ax[2].hist(mean_ipr_eig/np.mean(ipr_eig_ori),15,facecolor='r',alpha=0.5)
		# ax[1].hist(mean_ipr_eig/np.mean(ipr_eig_ori[ind_pickup_ori]),15,facecolor='r',alpha=0.5)
		# ax[1].set_xlabel('normalized mean ipr eig')
		# #ax[2].set_xlim([0,1])
		# ax[1].axvline(1)
		
		#---------------------------------------------------------------------------------
		# plot IPR for shuffled FLNs-II
		#---------------------------------------------------------------------------------
		mean_ipr_eig=np.zeros(p['n_area']+1)
		# mean_ipr_green=np.zeros(p['n_area']+1)
	
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
		# plt.xticks(np.arange(p['n_area']+1),p['areas'],rotation=90)
		#plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
		plt.ylabel('eigvector IPR')
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
		
		# #---------------------------------------------------------------------------------
		# # plot the distribution of complex eigenvalues for shuffled FLNs
		# #---------------------------------------------------------------------------------
		plt.figure(figsize=(5,5)) 
		plt.scatter(np.real(eigVals_complex), np.imag(eigVals_complex))
		plt.scatter(np.real(eigVals_complex_ori), np.imag(eigVals_complex_ori))
		plt.title('Distribution of complex eigenvalues')
		plt.xlabel('Real')
		plt.ylabel('Imag')
		
		plt.figure(figsize=(5,5)) 
		plt.scatter(np.real(eigVals_total), np.imag(eigVals_total))
		plt.scatter(np.real(eigVals_ori), np.imag(eigVals_ori))
		plt.title('Distribution of complex eigenvalues')
		plt.xlabel('Real')
		plt.ylabel('Imag')
	
	# #---------------------------------------------------------------------------------
	# # plot time constants for shuffled FLNs
	# #---------------------------------------------------------------------------------
	# mean_tau=np.zeros(p['n_area'])
	# top_90=np.zeros(p['n_area'])
	# bottom_10=np.zeros(p['n_area'])
	# top_95=np.zeros(p['n_area'])
	# bottom_5=np.zeros(p['n_area'])
	 
	# plt.figure(figsize=(5,5))		   
	# ax = plt.axes()
	# for k in np.arange(p['n_area']):
	#	  mean_tau[k]=np.mean(tau_shuffled[k,:])
	#	  sort_tau=np.sort(tau_shuffled[k,:])
	#	  top_90[k]=sort_tau[int(0.9*n_trial)]
	#	  bottom_10[k]=sort_tau[int(0.1*n_trial)]
	#	  top_95[k]=sort_tau[int(0.95*n_trial)]
	#	  bottom_5[k]=sort_tau[int(0.05*n_trial)]
		
	#	  plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
	#	  plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
		
		
	# plt.plot(np.arange(p['n_area']),mean_tau,'.b',markersize=10)	  
	# plt.plot(np.arange(p['n_area']),top_90,'.b',markersize=8,alpha=0.3)  
	# plt.plot(np.arange(p['n_area']),bottom_10,'.b',markersize=8,alpha=0.3)  
	# plt.plot(np.arange(p['n_area']),top_95,'.b',markersize=6,alpha=0.1)  
	# plt.plot(np.arange(p['n_area']),bottom_5,'.b',markersize=6,alpha=0.1)	 
	# plt.plot(np.arange(p['n_area']),tau_ori,'.k',markersize=10)
	
	# plt.xticks(np.arange(p['n_area']),p['areas'],rotation=90)
	# plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
	# plt.ylabel('$T_{delay}$ (ms)')
	# ax.spines['top'].set_visible(False)
	# ax.spines['right'].set_visible(False)
	
	# return record_fln_shuffled, sym_idx_lst, mean_ipr_total, num_imag_eig, ipr_eig_shuffled, ipr_hi_corr
	return record_fln_shuffled, ipr_eig_shuffled, ipr_eig_pos_shuffled, ipr_eig_neg_shuffled, ipr_weight_shuffled


def role_of_connection_area_by_shuffling_FLN(p_t,plot_Flag=1, MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0, n_trial=1000):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	
	if MACAQUE_CASE:
		hi_threshold = 0.
	else:
		hi_threshold = 0.
	# tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)
	[theta_area_ori, Green_coef_ori, eigVals_ori] = time_constant_area_shuffle_fln(p, p['fln_mat'])

	# n_trial=1000
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	Green_coef_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	theta_area_shuffled=np.zeros((p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))
	mean_theta_area=np.zeros(n_trial)
	
	for j in np.arange(n_trial):   
		print('n_trial=',j)
		
		max_eigval=1
		while max_eigval>0:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=pf.matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			# print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		# sym_idx_lst[j]=matrix_symmetric_index(fln_shuffled)
		p['fln_mat']=fln_shuffled
		[theta_area_shuffled_j, Green_coef_shuffled_j, eigVals_shuffled_j] \
			= time_constant_area_shuffle_fln(p, p['fln_mat'])
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		theta_area_shuffled[:, j], eigVals_shuffled[:, j]=theta_area_shuffled_j, np.real(eigVals_shuffled_j)
		Green_coef_shuffled[:, :, j] = Green_coef_shuffled_j
		# ind_pickup_j = [True]*p['n_area']
		mean_theta_area[j]=np.mean(theta_area_shuffled[:,j])
	
	mean_theta_total = mean_theta_area
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	if plot_Flag:
		plt.figure(figsize=(16, 9))		   
		ax = plt.axes()
		plt.hist(mean_theta_total,15,facecolor='r',alpha=0.5)
		plt.xlabel('mean theta (total)')
		#ax[0].set_xlim([0,1])
		plt.axvline(np.mean(theta_area_ori))
		
		#---------------------------------------------------------------------------------
		# plot IPR for shuffled FLNs-II
		#---------------------------------------------------------------------------------
		mean_theta_area=np.zeros(p['n_area']+1)
		# mean_ipr_green=np.zeros(p['n_area']+1)
	
		top_90=np.zeros(p['n_area']+1)
		bottom_10=np.zeros(p['n_area']+1)
		top_95=np.zeros(p['n_area']+1)
		bottom_5=np.zeros(p['n_area']+1)
		 
		plt.figure(figsize=(16, 9))		   
		ax = plt.axes()
		for k in np.arange(p['n_area']):
			mean_theta_area[k]=np.mean(theta_area_shuffled[k,:])
			sort_theta_area=np.sort(theta_area_shuffled[k,:])
			top_90[k]=sort_theta_area[int(0.9*n_trial)]
			bottom_10[k]=sort_theta_area[int(0.1*n_trial)]
			top_95[k]=sort_theta_area[int(0.95*n_trial)]
			bottom_5[k]=sort_theta_area[int(0.05*n_trial)]
			
			plt.vlines(k, bottom_10[k], top_90[k],color="blue",alpha=0.3)
			plt.vlines(k, bottom_5[k], top_95[k],color="blue",alpha=0.1)
		
		area_avg_theta=np.mean(theta_area_shuffled,0)
		mean_theta_area[p['n_area']]=np.mean(area_avg_theta)
		sort_area_avg_theta=np.sort(area_avg_theta)
		top_90[p['n_area']]=sort_area_avg_theta[int(0.9*n_trial)]
		bottom_10[p['n_area']]=sort_area_avg_theta[int(0.1*n_trial)]
		top_95[p['n_area']]=sort_area_avg_theta[int(0.95*n_trial)]
		bottom_5[p['n_area']]=sort_area_avg_theta[int(0.05*n_trial)]
		
		plt.vlines(p['n_area'], bottom_10[p['n_area']], top_90[p['n_area']],color="blue",alpha=0.3)
		plt.vlines(p['n_area'], bottom_5[p['n_area']], top_95[p['n_area']],color="blue",alpha=0.1)
			
		plt.plot(np.arange(p['n_area']+1),mean_theta_area,'.b',markersize=10)	 
		plt.plot(np.arange(p['n_area']+1),top_90,'.b',markersize=8,alpha=0.3)  
		plt.plot(np.arange(p['n_area']+1),bottom_10,'.b',markersize=8,alpha=0.3)  
		plt.plot(np.arange(p['n_area']+1),top_95,'.b',markersize=6,alpha=0.1)  
		plt.plot(np.arange(p['n_area']+1),bottom_5,'.b',markersize=6,alpha=0.1)	 
		theta_area_ori=np.append(theta_area_ori,np.mean(theta_area_ori))
		plt.plot(np.arange(p['n_area']+1),theta_area_ori,'.k',markersize=10)
		
		p['areas'].append('all')
		plt.xticks(np.arange(p['n_area']+1),p['areas'],rotation=90)
		#plt.yticks([100,1000,10000],['100 ms','1 s','10 s'],rotation=0)
		plt.ylabel('Theta')
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		
		fig, ax = plt.subplots(figsize=(16, 9))
		for i in range(0,p['n_area']):
			 ax.plot([i], [theta_area_ori[i]], '*')
			 ax.violinplot(dataset=theta_area_shuffled[i],positions=[i])
		plt.xticks(np.arange(p['n_area']),p['areas'][:-1],rotation=90)
	# return record_fln_shuffled, sym_idx_lst, mean_ipr_total, num_imag_eig, ipr_eig_shuffled, ipr_hi_corr
	return record_fln_shuffled, theta_area_shuffled, Green_coef_shuffled


def role_of_connection_by_spatial_distribution(p_t,MACAQUE_CASE=1,LINEAR_HIER=0):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	
	tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)

	[ipr_eig_ori,eigVals_ori]=time_constant_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)
	
	n_trial=1000
	lambda_lst = np.linspace(0, 1, n_trial)
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))

	
	mean_ipr_eig=np.zeros(n_trial)
	mean_ipr_eig_r=np.zeros(n_trial)
	mean_ipr_eig_i=np.zeros(n_trial)
	num_imag_eig=np.zeros(n_trial)
	
	eigVals_complex=np.array([])
	eigVals_total=np.array([])
	
	for j in np.arange(n_trial):   
		print('n_trial=',j)
		lambda_j = lambda_lst[j]
		
		max_eigval=1
		while max_eigval>-1e-5:
		# while max_eigval>-1e-4:	#not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=lambda_j*p_t['fln_mat_sort'] + (1-lambda_j)*p_t['fln_mat']
			max_eigval = -1
			# max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		p['fln_mat']=fln_shuffled
		ipr_eig_shuffled_j, eigVals_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled)
		# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
		eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
		# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
		ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
		eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
		eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
		# ind_pickup_j = [True]*p['n_area']
		mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
		mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
		mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
		num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
	
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	fig,ax=plt.subplots(1,4,figsize=(40,8)) 
	ax[0].plot(lambda_lst, mean_ipr_eig)
	ax[0].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	# ax[0].axvline(np.mean(ipr_eig_ori))
	
	
	ax[1].plot(lambda_lst, mean_ipr_eig_i)
	ax[1].set_xlabel('mean ipr eig (complex)')
	# ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
	
	ax[2].plot(lambda_lst, mean_ipr_eig_r)
	ax[2].set_xlabel('mean ipr eig (real)')
	# ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
	
	ax[3].plot(lambda_lst, num_imag_eig)
	ax[3].set_xlabel('number of complex eig')
	# ax[3].axvline(num_imag_eig_ori)
	
	return 

def role_of_connection_by_fln_strength(p_t,MACAQUE_CASE=1,LINEAR_HIER=0):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	
	tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)

	[ipr_eig_ori,eigVals_ori]=time_constant_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)
	
	n_trial=1000
	lambda_lst = np.linspace(0, 1, n_trial)
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))

	
	mean_ipr_eig=np.zeros(n_trial)
	mean_ipr_eig_r=np.zeros(n_trial)
	mean_ipr_eig_i=np.zeros(n_trial)
	num_imag_eig=np.zeros(n_trial)
	
	eigVals_complex=np.array([])
	eigVals_total=np.array([])
	
	for j in np.arange(n_trial):   
		lambda_j = lambda_lst[j]
		print('n_trial=',j)
		# print('lambda_j=', lambda_j)
		
		max_eigval=1
		while max_eigval>-1e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=lambda_j*p_t['fln_mat']
			# max_eigval = -1
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		p['fln_mat']=fln_shuffled
		ipr_eig_shuffled_j, eigVals_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled)
		# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
		eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
		# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
		ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
		eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
		eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
		# ind_pickup_j = [True]*p['n_area']
		mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
		mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
		mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
		num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
	
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	fig,ax=plt.subplots(1,4,figsize=(40,8)) 
	ax[0].plot(lambda_lst, mean_ipr_eig)
	ax[0].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	# ax[0].axvline(np.mean(ipr_eig_ori))
	
	
	ax[1].plot(lambda_lst, mean_ipr_eig_i)
	ax[1].set_xlabel('mean ipr eig (complex)')
	# ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
	
	ax[2].plot(lambda_lst, mean_ipr_eig_r)
	ax[2].set_xlabel('mean ipr eig (real)')
	# ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
	
	ax[3].plot(lambda_lst, num_imag_eig)
	ax[3].set_xlabel('number of complex eig')
	# ax[3].axvline(num_imag_eig_ori)
	
	return record_fln_shuffled, eigVals_shuffled


def role_of_connection_by_large_fln(p_t,MACAQUE_CASE=1,LINEAR_HIER=0):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	fln_lst = np.sort(fln_mat[fln_mat > 0])
	
	tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)

	[ipr_eig_ori,eigVals_ori]=time_constant_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)
	
	# n_trial=1000
	# lambda_lst = np.linspace(0, 1, n_trial, endpoint=False)
	n_trial = len(fln_lst)
	lambda_lst = range(n_trial)
	tau_shuffled=np.zeros((p['n_area'],n_trial))
	
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
	eigVals_shuffled=np.zeros((p['n_area'],n_trial))

	
	mean_ipr_eig=np.zeros(n_trial)
	mean_ipr_eig_r=np.zeros(n_trial)
	mean_ipr_eig_i=np.zeros(n_trial)
	num_imag_eig=np.zeros(n_trial)
	
	eigVals_complex=np.array([])
	eigVals_total=np.array([])
	
	for j in np.arange(n_trial):   
		lambda_j = lambda_lst[j]
		thr_j = fln_lst[lambda_j]
		# thr_j = fln_lst[np.int(lambda_j * len(fln_lst))]
		print('n_trial=',j)
		print('thr_j=', thr_j)
		
		max_eigval=1
		while max_eigval>-1e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=p_t['fln_mat'] * (p_t['fln_mat'] > thr_j)
			# max_eigval = -1
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		p['fln_mat']=fln_shuffled
		ipr_eig_shuffled_j, eigVals_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled)
		# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
		ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
		eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
		# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
		ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
		eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
		eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
		# ind_pickup_j = [True]*p['n_area']
		mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
		mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
		mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
		num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
	
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	fig,ax=plt.subplots(1,4,figsize=(40,8)) 
	ax[0].plot(lambda_lst, mean_ipr_eig)
	ax[0].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	# ax[0].axvline(np.mean(ipr_eig_ori))
	
	
	ax[1].plot(lambda_lst, mean_ipr_eig_i)
	ax[1].set_xlabel('mean ipr eig (complex)')
	# ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
	
	ax[2].plot(lambda_lst, mean_ipr_eig_r)
	ax[2].set_xlabel('mean ipr eig (real)')
	# ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
	
	ax[3].plot(lambda_lst, num_imag_eig)
	ax[3].set_xlabel('number of complex eig')
	# ax[3].axvline(num_imag_eig_ori)
	
	return record_fln_shuffled, eigVals_shuffled

def role_of_connection_by_swap_fln(p_t,SHUFFLE_TYPE = 15, CASE_NUM=1,MACAQUE_CASE=1,LINEAR_HIER=0):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	fln_lst = np.sort(fln_mat[fln_mat > 0])
	
	# tau_ori=pf.run_stimulus_pulse_macaque(p,fln_mat)

	[ipr_eig_ori,eigVals_ori]=time_constant_shuffle_fln(p,fln_mat)
	eigVals_r_ori = np.real(eigVals_ori)
	eigVals_i_ori = np.imag(eigVals_ori)
	# ind_pickup_ori = np.logical_not((eigVals_r_ori < -0.005) * (eigVals_r_ori > -0.012))
	ind_pickup_ori_imag = np.abs(eigVals_i_ori) > 1e-8
	eigVals_complex_ori = eigVals_ori[ind_pickup_ori_imag]
	# ind_pickup_ori = [True]*p['n_area']
	num_imag_eig_ori = np.sum(ind_pickup_ori_imag)
	
	# n_trial=1000
	# lambda_lst = np.linspace(0, 1, n_trial, endpoint=False)
	n_trial = len(fln_lst)
	sym_index_mat=np.zeros([n_trial, CASE_NUM])
	dist_corr_mat=np.zeros([n_trial, CASE_NUM])
	mean_ipr_eig_mat=np.zeros([n_trial, CASE_NUM])					 
	for id_case in range(CASE_NUM):
		print('CASE_NUM=', id_case)
		lambda_lst = np.array(range(n_trial))
		tau_shuffled=np.zeros((p['n_area'],n_trial))
		
		record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
		ipr_eig_shuffled=np.zeros((p['n_area'],n_trial))
		eigVals_shuffled=np.zeros((p['n_area'],n_trial))
	
		
		mean_ipr_eig=np.zeros(n_trial)
		mean_ipr_eig_r=np.zeros(n_trial)
		mean_ipr_eig_i=np.zeros(n_trial)
		num_imag_eig=np.zeros(n_trial)
		sym_index_lst=np.zeros([n_trial, ])
		dist_corr_lst=np.zeros([n_trial, ])
		
		eigVals_complex=np.array([])
		eigVals_total=np.array([])
		
		fln_mat=pf.matrix_random_permutation(p_t,p_t['fln_mat'],SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
		nz_index = np.nonzero(fln_mat)
		fln_lst = fln_mat[fln_mat > 0]
		sort_index_fln = np.argsort(-fln_lst)
		x_index_sort = nz_index[0][sort_index_fln]
		y_index_sort = nz_index[1][sort_index_fln]
	
		fln_k = fln_mat.copy()
		N_total = len(fln_lst)
			
		for j in np.arange(n_trial):   
			lambda_j = lambda_lst[j]
			# thr_j = fln_lst[np.int(lambda_j * len(fln_lst))]
			# print('n_trial=',j)
			
			max_eigval=1
			while max_eigval>-1e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
				fln_k = fln_k + 0.0
				pk = np.random.choice(N_total-lambda_j)
				temp = fln_k[x_index_sort[N_total-lambda_j-1], y_index_sort[N_total-lambda_j-1]]
				fln_k[x_index_sort[N_total-lambda_j-1], y_index_sort[N_total-lambda_j-1]] = fln_k[x_index_sort[pk], y_index_sort[pk]]
				fln_k[x_index_sort[pk], y_index_sort[pk]] = temp
				fln_shuffled = fln_k.copy()
				max_eigval = -1
				# max_eigval,_=pf.unstability_detection(p,fln_shuffled)
				# print('max_eigval=',max_eigval)
			
			# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
			record_fln_shuffled[:,:,j]=fln_shuffled
			p['fln_mat']=fln_shuffled
			ipr_eig_shuffled_j, eigVals_shuffled_j=time_constant_shuffle_fln(p,fln_shuffled)
			# ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=time_constant_shuffle_fln(p,fln_shuffled)
			ipr_eig_shuffled[:,j], eigVals_shuffled[:, j]=ipr_eig_shuffled_j, np.real(eigVals_shuffled_j)
			eigVals_shuffled_r_j = np.real(eigVals_shuffled_j)
			# ind_pickup_j = np.logical_not((eigVals_shuffled_r_j < -0.005) * (eigVals_shuffled_r_j > -0.012))
			ind_pickup_j_imag = np.abs(np.imag(eigVals_shuffled_j)) > 1e-8
			eigVals_complex=np.concatenate((eigVals_complex, eigVals_shuffled_j[ind_pickup_j_imag]))
			eigVals_total=np.concatenate((eigVals_total, eigVals_shuffled_j))
			# ind_pickup_j = [True]*p['n_area']
			mean_ipr_eig[j]=np.mean(ipr_eig_shuffled[:,j])
			mean_ipr_eig_i[j]=np.mean(ipr_eig_shuffled_j[ind_pickup_j_imag])
			mean_ipr_eig_r[j]=np.mean(ipr_eig_shuffled_j[np.logical_not(ind_pickup_j_imag)])
			num_imag_eig[j]=np.sum(np.abs(np.imag(eigVals_shuffled_j)) > 1e-8)
			sym_index_lst[j] = matrix_symmetric_index(fln_shuffled)
			dist_corr_lst[j] = dist_corr(p, fln_shuffled)
		sym_index_mat[:, id_case] = sym_index_lst
		mean_ipr_eig_mat[:, id_case] = mean_ipr_eig
		dist_corr_mat[:, id_case] = dist_corr_lst
	
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	fig,ax=plt.subplots(1,4,figsize=(40,8)) 
	ax[0].plot(lambda_lst, mean_ipr_eig)
	ax[0].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	# ax[0].axvline(np.mean(ipr_eig_ori))
	
	
	ax[1].plot(lambda_lst, mean_ipr_eig_i)
	ax[1].set_xlabel('mean ipr eig (complex)')
	# ax[1].axvline(np.mean(ipr_eig_ori[ind_pickup_ori_imag]))
	
	ax[2].plot(lambda_lst, mean_ipr_eig_r)
	ax[2].set_xlabel('mean ipr eig (real)')
	# ax[2].axvline(np.mean(ipr_eig_ori[np.logical_not(ind_pickup_ori_imag)]))
	
	ax[3].plot(lambda_lst, num_imag_eig)
	ax[3].set_xlabel('number of complex eig')
	# ax[3].axvline(num_imag_eig_ori)
	
	fig2=plt.figure(figsize=(20,8))
	plt.scatter(sym_index_lst, -dist_corr_lst)
	plt.scatter(matrix_symmetric_index(p_t['fln_mat']), np.mean(ipr_eig_ori))
	plt.xlabel('Sym Index')
	plt.ylabel('Mean IPR')
	
	fig3=plt.figure(figsize=(20,8))
	plt.scatter(lambda_lst, sym_index_lst)
	plt.plot(lambda_lst, matrix_symmetric_index(p_t['fln_mat']) + 0 * lambda_lst, '--', alpha=0.2)
	plt.ylabel('Sym Index')
	plt.xlabel('Iter num')
	
	fig4=plt.figure(figsize=(20,8))
	plt.scatter(lambda_lst, -dist_corr_lst)
	plt.plot(lambda_lst, dist_corr(p_t, p_t['fln_mat']) + 0 * lambda_lst, '--', alpha=0.2)
	plt.ylabel('Sym Index')
	plt.xlabel('Iter num')
	
	return sym_index_mat, dist_corr_mat, mean_ipr_eig_mat

def role_of_connection_by_symmetric_fln(p_t,MACAQUE_CASE=1,LINEAR_HIER=0,SHUFFLE_TYPE=0,fln_thr = 0.0075):
	
	#if SHUFFLE_TYPE==0:  #all elements are randomly permuted 
	#if SHUFFLE_TYPE==1:  #only permute the nonzero elements
	
	p=p_t.copy()
	fln_mat=p['fln_mat'].copy()
	sym_index_ori = matrix_symmetric_index(fln_mat)
	fln_ori_large = fln_mat * (fln_mat > fln_thr)
	sym_index_ori_large_fln = matrix_symmetric_index(fln_ori_large)
	
	n_trial=1000
	sym_index_lst=np.zeros([n_trial, ])
	sym_index_lst_large_fln=np.zeros([n_trial, ])
	record_fln_shuffled=np.zeros((p['n_area'],p['n_area'],n_trial))
	
	for j in np.arange(n_trial):   
		print('n_trial=',j)
		
		max_eigval=1
		while max_eigval>-1e-4:	  #not(max_eigval<-5e-4 and max_eigval>-6.5e-4):#-5e-4:
			fln_shuffled=pf.matrix_random_permutation(p,fln_mat,SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE)
			max_eigval,_=pf.unstability_detection(p,fln_shuffled)
			print('max_eigval=',max_eigval)
		
		# tau_shuffled[:,j]=pf.run_stimulus_pulse_macaque(p,fln_shuffled)
		record_fln_shuffled[:,:,j]=fln_shuffled
		sym_index_lst[j] = matrix_symmetric_index(fln_shuffled)
		fln_shuffled_large = fln_shuffled * (fln_shuffled > fln_thr)
		sym_index_lst_large_fln[j] = matrix_symmetric_index(fln_shuffled_large)
		
	#---------------------------------------------------------------------------------
	# plot IPR for shuffled FLNs-I 
	#---------------------------------------------------------------------------------
	fig,ax=plt.subplots(1,2,figsize=(40,8)) 
	ax[0].hist(sym_index_lst, bins=15)
	ax[0].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	ax[0].axvline(sym_index_ori)
	
	ax[1].hist(sym_index_lst_large_fln, bins=15)
	ax[1].set_xlabel('mean ipr eig (total)')
	#ax[0].set_xlim([0,1])
	ax[1].axvline(sym_index_ori_large_fln)
	
	return record_fln_shuffled, sym_index_lst

def matrix_symmetric_index(A):
	norm_A_sym = np.linalg.norm(A + A.T, ord='fro', axis=None)
	norm_A_antisym = np.linalg.norm(A - A.T, ord='fro', axis=None)
	sym_index = (norm_A_sym - norm_A_antisym) / (norm_A_sym + norm_A_antisym)
	return sym_index


#set the network parameters and generate the connectiivty matrix W in the linear dynamical system 
def genetate_net_connectivity_small_delta(p_t,MACAQUE_CASE=0,LINEAR_HIER=0,ZERO_HIER=0,IDENTICAL_HIER=0,LOCAL_IDENTICAL_HIERARCHY=0,LOCAL_LINEAR_HIERARCHY=0,LONG_RANGE_IDENTICAL_HIERARCHY=0, SHUFFLE_FLN=0,SHUFFLE_TYPE=0,ZERO_FLN=0,IDENTICAL_FLN=0,STRONG_GBA=0, DELETE_STRONG_LOOP=0,DELETE_CON_DIRECTION=0,GATING_PATHWAY=0,LONGRANGE_EI_ASYMMETRY=0,CONSENSUS_CASE=0):
	
	p=p_t.copy()
	
	if LINEAR_HIER+ZERO_HIER+IDENTICAL_HIER+LOCAL_IDENTICAL_HIERARCHY+LOCAL_LINEAR_HIERARCHY+LONG_RANGE_IDENTICAL_HIERARCHY>1 or SHUFFLE_FLN+IDENTICAL_FLN+ZERO_FLN>1:
		raise SystemExit('Conflict of network parameter setting!')
   
	#scale the hierarchy value linearly	   
	if LINEAR_HIER:
		p['hier_vals']=np.linspace(0,1,p['n_area'])
		print('LINEAR_HIER \n')
		
	#set the hierarchy value to be identical to zero 
	if ZERO_HIER:
		p['hier_vals']=np.zeros(len(p['areas']))
		print('ZERO_HIER \n')
		
	#set the hierarchy value to be identical to its mean
	if IDENTICAL_HIER:
		p['hier_vals']=np.ones(len(p['areas']))*np.mean(p['hier_vals'])
		print('IDENTICAL_HIER \n')
		
	#set the FLN value to be random
	if SHUFFLE_FLN:
		p['fln_mat']=matrix_random_permutation(p,p['fln_mat'],SHUFFLE_TYPE=SHUFFLE_TYPE,MACAQUE_CASE=MACAQUE_CASE) 
		print('SHUFFLE_FLN \n')
		
	#set the FLN value to be identical to its mean
	if IDENTICAL_FLN:
		#p['fln_mat']=np.ones_like(p['fln_mat'])*np.mean(p['fln_mat'])	 #all to all connected
		p['fln_mat'][p['fln_mat']>0]=np.mean(p['fln_mat'][p['fln_mat']>0])	#the topology remains the same but the weight is changed
		print('IDENTICAL_FLN \n')
		
	# disconnect all the inter-area connections 
	if ZERO_FLN:  
		p['fln_mat']=np.zeros_like(p['fln_mat'])
		print('ZERO_FLN \n')
			
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
#---------------------------------------------------------------------------------
# Network Parameters
#---------------------------------------------------------------------------------
	p['beta_exc'] = 0.066 # Hz/pA
	p['beta_inh'] = 0.351  # Hz/pA
	p['tau_exc'] = 20  # ms
	p['tau_inh'] = 10  # ms
	p['wEE'] = 24.4	 # pA/Hz
	p['wIE'] = 12.2	 # pA/Hz
	p['wEI'] = 19.7	 # pA/Hz
	p['wII'] = 12.5	 # pA/Hz 
	p['muEE']=33.7	 # pA/Hz  33.7#TEST TEST TEST 
	p['muIE'] = 25.5  # pA/Hz  25.3	 or smaller delta set 25.5
	p['eta'] = 0.68
	
	if CONSENSUS_CASE:
		if MACAQUE_CASE:
			p['muEE']=33.3	 # pA/Hz  33.7
			print('CONSENSUS_CASE=1, pay attention! Now muEE=',p['muEE'])
		else:
			p['muEE']=33.7	 # pA/Hz  33.7
			print('CONSENSUS_CASE=1, pay attention! Now muEE=',p['muEE'])
	else:
		print('CONSENSUS_CASE=0')	 
	
	#when the mini-cost network is reconstructed, the parameter muEE needs to be adjusted to avoid positive eigenvalue
	if SHUFFLE_TYPE==5:
		if MACAQUE_CASE:
			 p['muEE'] =  32.2	#33.7	# pA/Hz	  #CHANGED!!!!!!!
		else:
			 p['muEE'] = 33.2	#33.2	# pA/Hz	  #CHANGED!!!!!!!
		print('mini-cost network is reconstructed, pay attention! Now muEE=',p['muEE'])
				
	#strong GBA regime from joglekar etal 2018
	if STRONG_GBA:
		if MACAQUE_CASE:
			p['wEI'] = 25.2	 # pA/Hz
			p['muEE'] = 51.5  # pA/Hz
		else:
			p['wEI'] = 25.2	 # pA/Hz
			p['muEE'] = 48.5  # pA/Hz
		print('STRONG_GBA \n')
		
	p['exc_scale'] = (1+p['eta']*p['hier_vals'])
	
	local_EE =	p['beta_exc'] * p['wEE'] * p['exc_scale']
	local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
	local_IE =	p['beta_inh'] * p['wIE'] * p['exc_scale']
	local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
	
	fln_scaled = (p['exc_scale'] * p['fln_mat'].T).T
	
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
		p['exc_scale'] = (1+p['eta']*local_hier_vals)
		local_EE =	p['beta_exc'] * p['wEE'] * p['exc_scale']
		local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
		local_IE =	p['beta_inh'] * p['wIE'] * p['exc_scale']
		local_II = -p['beta_inh'] * p['wII'] *np.ones_like(local_EE)
		print('LOCAL_IDENTICAL_HIERARCHY \n')
		
	#keep long-range hierarchy graident, and set the local connection dependent of linear hierarchy
	if LOCAL_LINEAR_HIERARCHY:
		local_hier_vals=np.linspace(0,1,p['n_area'])
		p['exc_scale'] = (1+p['eta']*local_hier_vals)
		local_EE =	p['beta_exc'] * p['wEE'] * p['exc_scale']
		local_EI = -p['beta_exc'] * p['wEI'] *np.ones_like(local_EE)
		local_IE =	p['beta_inh'] * p['wIE'] * p['exc_scale']
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

def run_stimulus_reduced(p_t, W_t, VISUAL_INPUT=1,TOTAL_INPUT=0,T=6000,PULSE_INPUT=1,MACAQUE_CASE=1, CONSENSUS_CASE=0, stim_area='V1',plot_Flag=1):
	
	if VISUAL_INPUT:
		area_act = stim_area   #V1
	else:
		if MACAQUE_CASE:
			area_act='2'
		else:
			area_act = 'AuA1'
	print('Running network with stimulation to ' + area_act + '   PULSE_INPUT=' + str(PULSE_INPUT) + '   MACAQUE_CASE=' + str(MACAQUE_CASE))

	p = p_t.copy()
	W_EI = W_t.copy()
	A = W_EI[0:p['n_area'], 0:p['n_area']]
	B = W_EI[0:p['n_area'], p['n_area']::]
	C = W_EI[p['n_area']::, 0:p['n_area']]
	D = W_EI[p['n_area']::, p['n_area']::]
	W_reduced = np.linalg.inv(A+D)@(D@A - B@C)
	W_stim = np.linalg.inv(A+D)@D
	W_dot = np.linalg.inv(A+D)

	r_exc_base = 10
	r_exc_tgt = r_exc_base * np.ones(p['n_area'])
	I_bkg = - np.dot(W_reduced, r_exc_tgt)
	
	dt = 0.05   # ms
	if PULSE_INPUT:
		T = 2400
	else:				
		T = T
	t_plot = np.linspace(0, T, int(T/dt)+1)
	n_t = len(t_plot)
	
	# Set stimulus input
	I_stim_exc = np.zeros((n_t,p['n_area']))
	area_stim_idx = p['areas'].index(area_act) # Index of stimulated area
	
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
			I_stim_exc[:, area_stim_idx] = gaussian_noise(2,0.5,n_t)
	
	r_exc = np.zeros((n_t,p['n_area']))
	
	#---------------------------------------------------------------------------------
	# Initialization
	#---------------------------------------------------------------------------------
	# fI = lambda x : x*(x>0)
	fI = lambda x : x

	# Set activity to background firing
	r_exc[0] = r_exc_tgt
	
		#---------------------------------------------------------------------------------
		# Running the network
		#---------------------------------------------------------------------------------

	for i_t in range(1, n_t):	  
		I_exc = np.dot(W_stim, I_stim_exc[i_t])/p['tau_exc'] + I_bkg 
		d_r_exc = np.dot(W_reduced, r_exc[i_t-1]) + I_exc
		r_exc[i_t] = r_exc[i_t-1] + d_r_exc * dt
	
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
	
	if plot_Flag:
		area_idx_list=[-1]
		for name in area_name_list:
			area_idx_list=area_idx_list+[p['areas'].index(name)]
		#area_idx_list = [-1]+[p['areas'].index(name) for name in area_name_list]
		
		f, ax_list = plt.subplots(len(area_idx_list), sharex=True)
		
		clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
		c_color=0
		for ax, area_idx in zip(ax_list, area_idx_list):
			if area_idx < 0:
				y_plot = I_stim_exc[:, area_stim_idx].copy()
				z_plot = np.zeros_like(y_plot)
				txt = 'Input'

			else:
				y_plot = r_exc[:,area_idx].copy()
				txt = p['areas'][area_idx]

			if PULSE_INPUT:
				y_plot = y_plot - y_plot.min()
				# y_plot = y_plot - 10
				# z_plot = z_plot - z_plot.min()
				ax.plot(t_plot, y_plot,color='k')
				#ax.plot(t_plot, z_plot,'--',color='b')
			else:
				#ax.plot(t_plot, y_plot,color='r')
				ax.plot(t_plot[0:10000], y_plot[-1-10000:-1],color='r')
				# ax.plot(t_plot[0:10000], z_plot[-1-10000:-1],'--',color='b')
				
			# ax.plot(t_plot, y_plot,color=clist[0][c_color])
			# ax.plot(t_plot, z_plot,'--',color=clist[0][c_color])
			c_color=c_color+1
			ax.text(0.9, 0.6, txt, transform=ax.transAxes)

			if PULSE_INPUT:
				ax.set_yticks([0,y_plot.max()])
				# ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
				ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max())])
			ax.spines["right"].set_visible(False)
			ax.spines["top"].set_visible(False)
			#ax.xaxis.set_ticks_position('bottom')
			ax.yaxis.set_ticks_position('left')

		f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical')
		ax.set_xlabel('Time (ms)')	
		
		if PULSE_INPUT:
			clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
			plt.figure()
			posi_array=np.arange(np.size(time_idx))
			get_posi=posi_array[time_idx]
			get_posi=get_posi[-1]
			t_plot_cut = t_plot[get_posi:-1].copy()
			c_color=1
			for area_idx in area_idx_list:
				if area_idx < 0:
					continue
				else:
					y_plot_cut = r_exc[get_posi:-1,area_idx].copy()- r_exc[:,area_idx].min()
					y_plot_cut=y_plot_cut/y_plot_cut.max()
					plt.plot(t_plot_cut, y_plot_cut,color=clist[0][c_color])
					c_color=c_color+1
	
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
			ax.set_xticklabels(p['areas'],rotation=90,fontsize=18)
			#plt.yticks([10,100,1000],['10 ms','100 ms','1 s'],rotation=0)
			ax.set_ylabel('Decay time (ms)',fontsize=18)
			ax.spines['top'].set_visible(False)
			ax.spines['right'].set_visible(False)
		else:
			decay_time=np.zeros(p['n_area'])
		
		max_rate=np.max(r_exc-r_exc_base,axis=0)
		# network_graph_plot(p,max_rate,MACAQUE_CASE=MACAQUE_CASE)
		
		fig,ax=plt.subplots(figsize=(15,10))
		ax.plot(np.arange(len(p['areas'])), max_rate,'-o')
		ax.set_xticks(np.arange(len(p['areas'])))
		ax.set_xticklabels(p['areas'],rotation=90,fontsize=18)
		ax.set_yscale('log')
		ax.set_ylabel('Max Firing Rate',fontsize=18)
		# ax.set_xlabel('hierarchy values')
	
	return I_stim_exc, r_exc, area_stim_idx, dt, t_plot, decay_time
	# return I_stim_exc, r_exc, r_inh, area_stim_idx
	# return max_rate

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
	sigma=1
	B=np.zeros_like(W)
	
	for i in np.arange(p['n_area']):
		B[2*i,2*i]=sigma
	
	# B[0, 0] = sigma
	
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

def generate_func_connectivity_WEI(p_t,W_t):
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
	sigma=1
	B=np.zeros_like(W)
	
	for i in np.arange(p['n_area']):
		B[i,i]=sigma
	
	# B[0, 0] = sigma
	
	Q=inv_eigVecs.dot(B).dot(B.conj().T).dot(inv_eigVecs.conj().T)
	M=np.zeros_like(Q)
	for i in np.arange(2*p['n_area']):
		for j in np.arange(2*p['n_area']):
			M[i,j]=-Q[i,j]/(eigVals[i]+eigVals[j].conj())
	
	Cov_mat=eigVecs.dot(M).dot(eigVecs.conj().T)
	Cov_mat_E = Cov_mat[0:p['n_area'], 0:p['n_area']]
	Corr_mat_E=np.zeros((p['n_area'],p['n_area']))
	for i in np.arange(p['n_area']):
		for j in np.arange(p['n_area']):
			Corr_mat_E[i,j]=np.real(Cov_mat[i,j])/np.sqrt(Cov_mat[i,i]*Cov_mat[j,j])	  
	return Corr_mat_E, Cov_mat_E

def gen_conn_matrix(p_t):
	p = p_t.copy()
	conn = p['fln_mat'] > 0
	conn_sym = np.logical_and(conn, conn.T)
	p['conn'] = conn
	p['conn_sym'] = conn_sym
	return p


#define inverse participation ratio to meaure the non-spatial locality of a eigenvector	   
def IPR_est(vec_t,ind=4):
	vec=vec_t.copy()
	vec=np.abs(vec)
	ipr_ind=np.max(vec)/np.sum(vec)
	return ipr_ind


#define an metric to see how it fits with the expoential decay rule	  
def dist_corr(p_t, fln_t):
	p = p_t.copy()
	fln = fln_t.copy()
	dist_mat = p['dist_mat']
	dist_total = dist_mat[fln > 0]
	fln_total = fln[fln > 0]
	log_fln_total = np.log(fln_total)
	corr_temp = np.corrcoef([dist_total, log_fln_total])
	corr_coef = corr_temp[0, 1]
	return corr_coef

def theta_area(vec_t, eigVals,c=1,ind=1):
	vec = vec_t.copy()
	vals = eigVals.copy()
	vec = vec / np.linalg.norm(vec)
	vec = np.reshape(np.abs(vec), [-1, 1])
	vals = np.reshape(vals, [-1, 1])
	vec_mat = vec * vec.T
	time_const = 1/np.abs(-np.real(vals))
	diff_time_const_temp = np.abs(time_const - time_const.T)
	# diff_val_temp = np.abs(vals-vals.T)
	diff_time_const_temp[diff_time_const_temp == 0] = 1e4
	diff_reg = np.mean(np.min(diff_time_const_temp, 0))
	val_mat = np.exp(-c * np.abs(time_const - time_const.T) / diff_reg)
	M = np.sum((vec_mat**ind) * val_mat) / np.power(np.sum(vec**ind), 2)
	return M

def single_exp(x,a,b):
	return b*np.exp(-x/a)

def double_exp(x,a,b,c,d):
	return c*np.exp(-x/a)-d*np.exp(-x/b)

def gaussian_noise(mu,sigma,n_t):
	input_sig=np.zeros(n_t)
	for i in range(0,n_t):
		input_sig[i]=random.gauss(mu,sigma)
	return input_sig

def kappa_matrix(S):
	# eigVals, eigVecs = np.linalg.eig(S)
	# kappa = np.linalg.norm(eigVecs, 2) * np.linalg.norm(np.linalg.inv(eigVecs), 2)
	kappa = np.linalg.norm(S, 2) * np.linalg.norm(np.linalg.inv(S), 2)
	return kappa

def dep(S):
	T, Z = scipy.linalg.schur(S)
	Lamb = np.diag(np.diag(T))
	dep = np.sqrt(np.linalg.norm(S, 'fro')**2 - np.linalg.norm(Lamb, 'fro')**2)
	return dep

def Time_Local_Measure(p, norm_flag=0):
	p_t = p.copy()
	a0 = p_t['beta_exc'] / p_t['tau_exc'] * (p_t['wEE'] - 1/p_t['beta_exc'])
	a1 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEE'] * p_t['eta_local']
	a2 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['muEE']
	b = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEI']
	c0 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE']
	c1 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE'] * p_t['eta_inh_local']
	d = p_t['beta_inh'] / p_t['tau_inh'] * (p_t['wII'] + 1/p_t['beta_inh'])
	c2 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['muIE']
	m1 = (b*c0 - d*a0) / (a0-d)
	m2 = - ((b*c2 - d*a2) - a2 * m1) # / (b*c0 - d*a0)
	# m2 = M_SP + a2/(a0-d)
	m3 = ((b*c0 - d*a0) + (b*c1-d*a1)) / (a0-d+a1)
	if norm_flag:
		m2 /= (b*c0 - d*a0)
	return m2

def Time_Local_Measure_FULL(p, norm_flag=0):
	p_t = p.copy()
	a0 = p_t['beta_exc'] / p_t['tau_exc'] * (p_t['wEE'] - 1/p_t['beta_exc'])
	a1 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEE'] * p_t['eta_local']
	a2 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['muEE']
	b = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEI']
	c0 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE']
	c1 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE'] * p_t['eta_inh_local']
	d = p_t['beta_inh'] / p_t['tau_inh'] * (p_t['wII'] + 1/p_t['beta_inh'])
	c2 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['muIE']
	lamb_lst = (b*(c0+c1*p_t['hier_vals']) - d*(a0+a1*p_t['hier_vals'])) / ((a0+a1*p_t['hier_vals'])-d)
	m2_lst = - ((b*c2*(1+p_t['eta_inh']*p_t['hier_vals']) - d*a2*(1+p_t['eta']*p_t['hier_vals'])) \
			 - np.mean(lamb_lst)*a2*(1+p_t['eta']*p_t['hier_vals'])) # / (b*c0 - d*a0)
	# m2 = M_SP + a2/(a0-d)
	m3 = ((b*c0 - d*a0) + (b*c1-d*a1)) / (a0-d+a1)
	if norm_flag:
		m2_lst /= (b*(c0+c1*p_t['hier_vals']) - d*(a0+a1*p_t['hier_vals'])) 
	return m2_lst

def Time_Local_Measure_MAT(p, norm_flag=0):
	p_t = p.copy()
	a0 = p_t['beta_exc'] / p_t['tau_exc'] * (p_t['wEE'] - 1/p_t['beta_exc'])
	a1 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEE'] * p_t['eta_local']
	a2 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['muEE']
	b = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEI']
	c0 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE']
	c1 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE'] * p_t['eta_inh_local']
	d = p_t['beta_inh'] / p_t['tau_inh'] * (p_t['wII'] + 1/p_t['beta_inh'])
	c2 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['muIE']
	lamb_lst = (b*(c0+c1*p_t['hier_vals']) - d*(a0+a1*p_t['hier_vals'])) / ((a0+a1*p_t['hier_vals'])-d)
	lamb_lst = lamb_lst[:, np.newaxis]
	m2_mat = - ((b*c2*(1+p_t['eta_inh']*p_t['hier_vals']) - d*a2*(1+p_t['eta']*p_t['hier_vals'])) \
			 - lamb_lst*a2*(1+p_t['eta']*p_t['hier_vals'])) # / (b*c0 - d*a0)
	# m2 = M_SP + a2/(a0-d)
	m3 = ((b*c0 - d*a0) + (b*c1-d*a1)) / (a0-d+a1)
	if norm_flag:
		m2_mat /= (b*(c0+c1*p_t['hier_vals']) - d*(a0+a1*p_t['hier_vals'])) 
	return m2_mat



def Signal_Prop_Measure(p, norm_flag=0):
	p_t = p.copy()
	a0 = p_t['beta_exc'] / p_t['tau_exc'] * (p_t['wEE'] - 1/p_t['beta_exc'])
	b = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEI']
	c0 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE']
	d = p_t['beta_inh'] / p_t['tau_inh'] * (p_t['wII'] + 1/p_t['beta_inh'])
	a2 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['muEE']
	c2 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['muIE']
	m = - (b*c2 - d*a2) # / (b*c0 - d*a0)
	if norm_flag:
		m /= (b*c0 - d*a0)
	return m

#plot functional connectivity
def plot_func_connectivity(p_t,Corr_mat_E, fig, ax):
	
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
	
	# fig, ax = plt.subplots()
	f=ax.pcolormesh(Corr_mat_cut,cmap='hot')
	ax.set_title('functional connectivity')	
	# set original ticks and ticklabels
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_xticks(x[::2])
	ax.set_xticklabels(xticklabels_even, fontsize=15)
	ax.set_yticks(y[::2])
	ax.set_yticklabels(yticklabels_even, fontsize=15)
	ax.invert_yaxis()
	# rotate xticklabels to 90 degree
	plt.setp(ax.get_xticklabels(), rotation=90)
	
	# second x axis
	ax2 = ax.twiny()
	ax2.set_xlim(xlim)
	ax2.set_xticks(x[1::2])
	ax2.set_xticklabels(xticklabels_odd, fontsize=15)
	# rotate xticklabels to 90 degree
	plt.setp(ax2.get_xticklabels(), rotation=90)
	
	# second y axis
	ax3 = ax.twinx()
	ax3.set_ylim(ylim)
	ax3.set_yticks(y[1::2])
	ax3.set_yticklabels(yticklabels_odd, fontsize=15)
	ax3.invert_yaxis()
	
	cbar = fig.colorbar(f, ax=ax, pad=0.1)
	cbar.ax.tick_params(labelsize=20)  # set fontsize to 12
	# # fig.savefig('result/Functional_Connectivity.pdf')	

	# #---------------------------------------------------------------------------------
	# # correlation between functional connectivity and FLN
	# #--------------------------------------------------------------------------------- 
	# # FLN=p['fln_mat'] * p['sln_mat']
	# FLN=p['fln_mat']		
	# np.fill_diagonal(Corr_mat_E,0)
	# np.fill_diagonal(FLN,0)
	# corr_flat=Corr_mat_E.flatten()
	
	# # FLN = FLN * p['sln_mat']
	# fln_flat=FLN.flatten()
	# # corr_flat[np.where(fln_flat==0)]=1e-8
	# # fln_flat[np.where(fln_flat==0)]=1e-8
	
	# # pick_index = (fln_flat > 1e-6) * (corr_flat > 1e-4)
	# # fln_flat = fln_flat[pick_index]
	# # corr_flat= corr_flat[pick_index]
	
	# fig,ax=plt.subplots()
	# ax.scatter(fln_flat,corr_flat)	  
	# ax.set_xlim((1e-6,1e0))
	# ax.set_ylim((1e-4,1e0))
	# ax.set_xscale('log')
	# ax.set_yscale('log')
	# ax.set_xlabel('FLN')
	# ax.set_ylabel('Functional Connectivity')
	
	# ce=np.corrcoef(fln_flat,corr_flat)
	# ax.set_title('corrcoef='+str(ce[0,1])[:4])
	# print('corrcoef=',ce[0,1])
	# # fig.savefig('result/Coeff_VS_FLN.pdf')	

#plot functional connectivity
def plot_eigenvalues(p_t,eigVals, eigVecs, fig, ax, flag_sort=0):
	
	p=p_t.copy()
	
	eigVecs_cut=eigVecs.copy()
	# Corr_mat_cut[Corr_mat_E>0.3]=0.3
	# Corr_mat_cut[Corr_mat_E<0.01]=0
	
	x = np.arange(len(p['areas'])) # xticks
	y = np.arange(len(p['areas'])) # yticks
	xlim = (0,len(p['areas']))
	ylim = (0,len(p['areas']))
	
	tau = - 1 / np.real(eigVals)
	# xticklabels_odd  = tau[::2]

	if flag_sort:
		sort_id = np.argsort(p['hier_vals'])
	else:
		sort_id = np.arange(len(p['areas']))

	xticklabels_even = ['{:.2f}'.format(tau_k) for tau_k in tau[1::2]]
	yticklabels_odd=np.array(p['areas'])[sort_id][1::2]
	yticklabels_even=np.array(p['areas'])[sort_id][::2]
	
	# fig, ax = plt.subplots()
	f=ax.pcolormesh(np.abs(eigVecs_cut[sort_id]),cmap='hot')
	ax.set_title('Eigenvectors')	
	# set original ticks and ticklabels
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	if flag_sort == 1:
		ax.set_xticks(x)
		xticklabels_full = ['{:.2f}'.format(tau_k) for tau_k in tau]
		ax.set_xticklabels(xticklabels_full)
	else:
		ax.set_xticks(x[1::2])
		ax.set_xticklabels(xticklabels_even)
	ax.set_yticks(y[::2])
	ax.set_yticklabels(yticklabels_even)
	# ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
	ax.invert_yaxis()
	# rotate xticklabels to 90 degree
	plt.setp(ax.get_xticklabels(), rotation=90)
	
	# # second x axis
	# ax2 = ax.twiny()
	# ax2.set_xlim(xlim)
	# ax2.set_xticks(x[1::2])
	# ax2.set_xticklabels(xticklabels_odd)
	# # rotate xticklabels to 90 degree
	# plt.setp(ax2.get_xticklabels(), rotation=90)
	
	# second y axis
	ax3 = ax.twinx()
	ax3.set_ylim(ylim)
	ax3.set_yticks(y[1::2])
	ax3.set_yticklabels(yticklabels_odd)
	ax3.invert_yaxis()
	
	fig.colorbar(f,ax=ax,pad=0.1)


def plot_timescale(p_t, fig, ax,flag_sort=0):
	p = p_t.copy()
	W_EI = generate_W_EI_shuffle_fln(p.copy(),p['fln_mat'].copy())
	eigVals, eigVecs = np.linalg.eig(W_EI)
	coef, delay_time, acf_data, tau_s = theoretical_time_constant_input_at_all_areas(p,eigVecs,eigVals, plot_Flag=0)

	clist = plt.cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
	if flag_sort:
		sort_id = np.argsort(p['hier_vals'])
	else:
		sort_id = np.arange(len(p['areas']))
	ax.barh(np.arange(p['n_area']), delay_time[sort_id], height=1.0, align='center', color=clist[0])
	ax.set_xscale('log')
	ax.invert_xaxis()
	plt.xticks(fontsize=18)
	plt.xlabel('Timescale (ms)', fontsize=18)
	ylim = (0,len(p['areas']))
	if p['n_area'] < 40:
		ax.set_yticks(np.arange(p['n_area']))
		ax.set_yticklabels(np.array(p['areas'])[sort_id], fontsize=18)
		ax.invert_yaxis()
	else:
		y = np.arange(len(p['areas'])) # yticks
		yticklabels_odd=np.array(p['areas'])[sort_id][1::2]
		yticklabels_even=np.array(p['areas'])[sort_id][::2]
		ax.set_yticks(y[::2])
		ax.set_ylim(ylim)
		ax.set_yticklabels(yticklabels_even)
		ax.invert_yaxis()
		# second y axis
		ax3 = ax.twinx()
		ax3.set_ylim(ylim)
		ax3.set_yticks(y[1::2])
		ax3.set_yticklabels(yticklabels_odd)
		ax3.invert_yaxis()
	return delay_time[sort_id]


def plot_dynamics(p_t, VISUAL_INPUT=1,TOTAL_INPUT=0,T=1000,PULSE_INPUT=1, MACAQUE_CASE=0, flag_sort=1, flag_rate=0):

	p = p_t.copy()

	I_stim_exc, r_exc, r_inh, area_stim_idx, dt, t_plot, decay_time, max_rate \
	= pf.run_stimulus(p,VISUAL_INPUT=VISUAL_INPUT,TOTAL_INPUT=TOTAL_INPUT,T=T,PULSE_INPUT=PULSE_INPUT,
		MACAQUE_CASE=MACAQUE_CASE,GATING_PATHWAY=0,CONSENSUS_CASE=0,plot_Flag=0)

	if MACAQUE_CASE:
		if VISUAL_INPUT:
			area_name_list = ['V1','V4','8m','8l','TEO','7A','9/46d','TEpd','24c']
			# area_name_list = p['areas']
			# area_name_list = ['V1','V4','8m', 'TEO','STPc', '10','TEpd', 'ProM','24c']
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
	
	clist = cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
	f, ax_list = plt.subplots(len(area_idx_list), sharex=True, figsize=(8, 16))
	
	# clist = cm.get_cmap(plt.get_cmap('Blues'))(np.linspace(0.0, 1.0, len(area_idx_list)))[np.newaxis, :, :3]
	c_color=0
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
		else:
			ax.plot(t_plot[0:10000], y_plot[-1-10000:-1],color='k', linewidth=1)
		c_color=c_color+1
		# ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=18)

		if PULSE_INPUT:
			ax.set_yticks([0,y_plot.max()])
			# ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max()),'{:0.4f}'.format(z_plot.max())])
			ax.set_yticklabels([0,'{:0.4f}'.format(y_plot.max())], fontsize=18)
		ax.text(0.9, 0.6, txt, transform=ax.transAxes, size=20)
		# Hide the top and right spines
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Make the bottom and left spines thicker
		ax.spines['bottom'].set_linewidth(2)
		ax.spines['left'].set_linewidth(2)

	f.text(0.01, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=20)
	ax.set_xlabel('Time (ms)', fontsize=20) 

	# 	ax.spines["right"].set_visible(False)
	# 	ax.spines["top"].set_visible(False)
	# 	# ax.xaxis.set_ticks_position('bottom')
	# 	ax.yaxis.set_ticks_position('left')
	# 	plt.xticks(fontsize=18)

	# f.text(0.9, 0.5, 'Change in firing rate (Hz)', va='center', rotation='vertical', size=18)
	# ax.set_xlabel('Time (ms)', fontsize=18)
	# # ax.set_ylabel('Change in firing rate (Hz)', fontsize=18)


	if flag_sort:
		sort_id = np.argsort(p['hier_vals'])
	else:
		sort_id = np.arange(len(p['areas']))

	f2, ax2 = plt.subplots(1,1,figsize=(5, 10))
	clist = plt.cm.get_cmap(plt.get_cmap('rainbow'))(np.linspace(0.0, 1.0, p['n_area']))[np.newaxis, :, :3]
	ax2.barh(np.arange(p['n_area']), max_rate[sort_id], height=1.0, align='center', color=clist[0])
	ax2.set_xscale('log')
	plt.xticks(fontsize=18)
	plt.xlabel('Max Firing Rate (Hz)', fontsize=18)
	ylim = (0,len(p['areas']))
	if p['n_area'] < 40:
		ax2.set_yticks(np.arange(p['n_area']))
		ax2.set_yticklabels(np.array(p['areas'])[sort_id], fontsize=18)
		ax2.invert_yaxis()
	else:
		y = np.arange(len(p['areas'])) # yticks
		yticklabels_odd=np.array(p['areas'])[sort_id][1::2]
		yticklabels_even=np.array(p['areas'])[sort_id][::2]
		ax2.set_yticks(y[::2])
		ax2.set_ylim(ylim)
		ax2.set_yticklabels(yticklabels_even)
		ax2.invert_yaxis()
		# second y axis
		ax3 = ax2.twinx()
		ax3.set_ylim(ylim)
		ax3.set_yticks(y[1::2])
		ax3.set_yticklabels(yticklabels_odd)
		ax3.invert_yaxis()
	if flag_rate:
		return f, ax, f2, ax2, max_rate[sort_id]
	else:
		return f, ax, f2, ax2

def parameter_recover_2(p, M_TL, M_SP):
	p_t = p.copy()
	a0 = p_t['beta_exc'] / p_t['tau_exc'] * (p_t['wEE'] - 1/p_t['beta_exc'])
	a1 = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEE'] * p_t['eta_local']
	b = p_t['beta_exc'] / p_t['tau_exc'] * p_t['wEI']
	c0 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE']
	c1 = p_t['beta_inh'] / p_t['tau_inh'] * p_t['wIE'] * p_t['eta_inh_local']
	d = p_t['beta_inh'] / p_t['tau_inh'] * (p_t['wII'] + 1/p_t['beta_inh'])
	# lamb = (b*c0 - d*a0) / (a0-d)
	# Updated
	lamb_lst = (b*(c0+c1*p_t['hier_vals']) - d*(a0+a1*p_t['hier_vals'])) / ((a0+a1*p_t['hier_vals'])-d)
	lamb = np.mean(lamb_lst)
	# M_TL = M_SP + lamb * beta_E/tau_E * muEE
	# lamb = -0.02
	muEE = (M_TL - M_SP) / (lamb * p_t['beta_exc']) * p_t['tau_exc']
	# M_SP = - beta_E*beta_I / (tau_E*tau_I) * (wEI muIE - (wII+1/beta_I)*muEE) 
	delta = - M_SP * p_t['tau_exc']*p_t['tau_inh'] / (p_t['beta_exc']*p_t['beta_inh'])
	muIE = (delta + (p_t['wII']+1/p_t['beta_inh'])*muEE) / p_t['wEI']
	return muEE, muIE