# -*- coding: utf-8 -*-

"""
Noise estimation based on Rye and Hardesty 1993
"""
import os
cd=os.path.dirname(__file__)
import sys
sys.path.append('C:/Users/SLETIZIA/OneDrive - NREL/Desktop/PostDoc/utils')
import utils as utl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.size'] = 16


#%% Inputs
source_noise=os.path.join(cd,'data','sc1.lidar_z01-halo_xrp_user2_z01.a0.ppr=2985.dr=48.snr.noise.cutoff40.xlsx')
SNRs=np.arange(-30,1,1)#[dB] SNR grid
lambda0=1.534*10**-6#[m] laser wavelenght
f_sample=100*10**6#[Hz] sampling frequency

f1=0#mean of spectral peak
f2=0.1#spectral width
M=32#number of FFT points
N=1000#accumulation
L=10000#number of MC samples

#%% Initialization
f=np.arange(M+1)/M-0.5#normalized frequencies

Data=pd.read_excel(source_noise)

#%% Main
rws=[]
rws_max=[]
for SNR in SNRs:
    delta=10**(SNR/10)
    phi=delta/(2*np.pi)**0.5/f2*np.exp(-(f-f1)**2/(2*f2**2))+1#mean spectrum
    x=np.random.gamma(shape=np.zeros((L,M+1))+N, scale=np.tile(phi/N,(L,1)))#random accumulated spectra
    
    #ML peak estimator
    Lambda=[]
    for f_ml in f:
        phi_ml=delta/(2*np.pi)**0.5/f2*np.exp(-(f-f_ml)**2/(2*f2**2))+1
        phi_ml=phi_ml/np.sum(phi_ml)
        Lambda=utl.vstack(Lambda,np.sum(x*np.tile(1/phi_ml,(L,1)),axis=1))
    rws=utl.vstack(rws,f[np.argmin(Lambda,axis=0)]*f_sample/2*lambda0)
    
    #maximum peak estimator
    rws_max=utl.vstack(rws_max,f[np.argmax(x,axis=1)]*f_sample/2*lambda0)

    print(SNR)
    
delta=10**(SNRs/10)
std_cr=(4*np.pi**0.5*f2**3/(N*M*delta**2)*(1+0.16*delta/f2)**2)**0.5*f_sample/2*lambda0
    
#%% Plots
plt.figure(figsize=(18,8))
ax=plt.subplot(1,2,1)
plt.plot(SNRs,np.mean(rws-f1*f_sample/2*lambda0,axis=1),'r',label='ML')
plt.plot(SNRs,np.mean(rws_max-f1*f_sample/2*lambda0,axis=1),'k',label='Max')
ax.grid(visible=True)
plt.xlabel(r'SNR [dB]')
plt.ylabel(r'Bias of $u_{LOS}$ [m s$^{-1}$]')

ax=plt.subplot(1,2,2)
plt.semilogy(SNRs,np.std(rws,axis=1),'r',label='ML')
plt.semilogy(SNRs,np.std(rws_max,axis=1),'k',label='Max')
plt.semilogy(Data['SNR [dB]'],Data['Noise StDev [m/s]'],'.g',markersize=10,label='Exp.')
plt.plot(SNRs,std_cr,'b',label='CRLB')
ax.grid(visible=True)
plt.ylim([10**-2,25])
plt.xlabel(r'SNR [dB]')
plt.ylabel(r'St. Dev. of $u_{LOS}$ [m s$^{-1}$]')
plt.legend()

