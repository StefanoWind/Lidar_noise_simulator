# -*- coding: utf-8 -*-

"""
Draw pdf of lidar noise based on Rye and Hardesty 1993
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs
SNRs=np.arange(0,-31,-2)#[dB] SNR grid
Ms=[16,32,48]#number of FFT points
Ns=[100,1000,10000]
lambda0=1.534*10**-6#[m] laser wavelenght
f_sample=100*10**6#[Hz] sampling frequency

f1=0#mean of spectral peak
f2=0.1#spectral width

L=10000#number of MC samples
std_low=0.5
std_sat=0.9

colors={'low':'g','enh':'y','sat':'r'}

#%% Initialization
bins_rws=np.linspace(-0.5*f_sample/2*lambda0,0.5*f_sample/2*lambda0,50)

#%% Main

# fig=plt.figure(figsize=(18,8))
# N=Ns[int(np.floor(len(Ns)/2))]
# ctr=1
# for M in Ms:
#     ax = fig.add_subplot(1,len(Ms),ctr, projection='3d')
    
#     f=np.arange(M+1)/M-0.5#normalized frequencies
#     for SNR in SNRs:
#         delta=10**(SNR/10)
#         phi=delta/(2*np.pi)**0.5/f2*np.exp(-(f-f1)**2/(2*f2**2))+1#mean spectrum
#         x=np.random.gamma(shape=np.zeros((L,M+1))+N, scale=np.tile(phi/N,(L,1)))#random accumulated spectra
        
#         #maximum peak estimator
#         rws=f[np.argmax(x,axis=1)]*f_sample/2*lambda0
        
#         hist = np.histogram(rws, bins=bins_rws)[0]
#         hist=hist/np.sum(hist)
#         ax.bar(utl.mid(bins_rws), hist, zs=SNR,width=10, zdir='y', alpha=0.8,color='k')
#         ax.view_init(40,-60)
#         ax.set_box_aspect([1,2,1])
#     ctr+=1


fig=plt.figure(figsize=(18,8))
M=Ms[int(np.floor(len(Ms)/2))]
f=np.arange(M+1)/M-0.5#normalized frequencies
ctr=1
for N in Ns:
    ax = fig.add_subplot(1,len(Ms),ctr, projection='3d')
    
    for SNR in SNRs:
        delta=10**(SNR/10)
        phi=delta/(2*np.pi)**0.5/f2*np.exp(-(f-f1)**2/(2*f2**2))+1#mean spectrum
        x=np.random.gamma(shape=np.zeros((L,M+1))+N, scale=np.tile(phi/N,(L,1)))#random accumulated spectra
        
        #maximum peak estimator
        rws=f[np.argmax(x,axis=1)]*f_sample/2*lambda0

        if np.std(rws)<std_low*(2*lambda0*f_sample/4)/12**0.5:
            regime='low'
        elif np.std(rws)>std_sat*(2*lambda0*f_sample/4)/12**0.5:
            regime='sat'
        else:
            regime='enh'
        hist = np.histogram(rws, bins=bins_rws)[0]
        hist=hist/np.sum(hist)
        ax.bar(utl.mid(bins_rws), hist, zs=SNR,width=3, zdir='y', alpha=0.75,color=colors[regime])
        ax.view_init(40,-60)
        ax.set_box_aspect([1,3,1])
        ax.set_zticklabels([])
        ax.set_xlabel(r'$u_{LOS}$ [m s$^{-1}$]')
        ax.set_yticks(np.arange(0,-31,-5))
        if ctr==len(Ns):
            ax.set_ylabel('\n \n \n SNR [dB]')
        plt.title(r'$N='+str(N)+'$')
    ctr+=1
plt.tight_layout()