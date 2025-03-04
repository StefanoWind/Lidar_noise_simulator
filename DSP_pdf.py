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

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16

#%% Inputs
SNRs=np.arange(0,-31,-2)#[dB] SNR grid
Ns=[100,1000,10000]# PPR
M=32#FFT points
lambda0=1.534*10**-6#[m] laser wavelenght
f_sample=100*10**6#[Hz] sampling frequency

f1=0#mean of spectral peak
f2=0.1#spectral width

L=10000#number of MC samples
std_low=0.1 #fraction of bandwide noise std to define low noise regime
std_sat=0.9 #fraction of bandwide noise std to define high noise regime

#graphics
colors={'low':'g','enh':'y','sat':'r'}

#%% Initialization
bins_rws=np.linspace(-(0.5+1/M/2)*f_sample/2*lambda0,(0.5+1/M/2)*f_sample/2*lambda0,M+2)

fig=plt.figure(figsize=(18,8))

#%% Main
f=np.arange(M+1)/M-0.5#normalized frequencies
ctr=1
for N in Ns:

    ax = fig.add_subplot(1,len(Ns),ctr, projection='3d')
    
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
        ax.bar(utl.mid(bins_rws),hist, zs=SNR,width=3, zdir='y', alpha=0.75,color=colors[regime])
        ax.view_init(40,-60)
        ax.set_box_aspect([1,3,2])
        ax.set_zlim([0,1])
        ax.set_xlabel(r'$u_{LOS}/u_{Nyquist}$')
        ax.set_xticks(np.arange(-1,1.1,0.5)*lambda0*f_sample/4)
        ax.set_yticks(np.arange(0,-31,-5))
        ax.set_xticklabels(np.arange(-1,1.1,0.5))
        ax.set_zticks([])
        ax.set_zticklabels([])
            
        if ctr==len(Ns):
            ax.set_ylabel('\n \n \n SNR [dB]')
        plt.title(r'$N='+str(N)+'$')

    # Make the panes (the background planes) white
    ax.xaxis.pane.set_facecolor('white')
    ax.yaxis.pane.set_facecolor('white')
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    
    # Make the grid lines white
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 1)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 1)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 1)
    
    # Make the axes lines white
    ax.zaxis.line.set_color('white')
    
    #redraw grid
    ylim=ax.get_ylim()
    for x in np.arange(-1,1.1,0.5)*lambda0*f_sample/4:
        plt.plot([x,x],[ylim[0],ylim[1]],color=(0,0,0,0.25),linewidth=1)
    xlim=ax.get_xlim()
    for y in np.arange(0,-31,-5):
        plt.plot([xlim[0],xlim[1]],[y,y],color=(0,0,0,0.25),linewidth=1)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    ctr+=1
plt.tight_layout()