# -*- coding: utf-8 -*-
'''
Analyze lidar noise curves
'''

import os
cd=os.path.dirname(__file__)
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import glob
import re
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
matplotlib.rcParams['font.size'] = 18
plt.close('all')

global Nyquist
global slope_lin
global snr_flat

#%% Inputs

#user
plot_theory=False

#dataset
source='data/nwtc.lidar.z01.nearest/*xlsx'
pattern=r"ppr=(\d+).*?dr=(\d+)"
fs=100*10**6#[Hz] sampling frequency 
c=299792458#[m/s] speed of light
Nyquist=39.14#[m/s] Nyquist velocity
slope_lin=-np.log(10)/10#[m/s/dB] slope of linear part fo noise curve
snr_flat=-30#[dB] lower limit of SNR
f2=5.5/(2*Nyquist)

#stats
max_unc=2#[m/s] max statistican uncertainty on noise
snr_max=-10#max SNR
min_N=10#minimum number of points

#graphics
snr_plot=np.arange(snr_flat,snr_max+0.1,0.1)
cmap = cm.get_cmap('viridis')

#%% Functions

def curve(snr,snr_lin,noise_lin):
    f=np.zeros_like(snr)+np.nan
    f_max=np.log(Nyquist*2/12**0.5)
    f_lin=np.log(noise_lin)
    
    f[snr<=snr_flat]=f_max
    f[snr>=snr_lin]=(snr[snr>=snr_lin]-snr_lin)*slope_lin+f_lin

    x=(snr[(snr>=snr_flat)*(snr<=snr_lin)]-snr_flat)/(snr_lin-snr_flat)
    f[(snr>=snr_flat)*(snr<=snr_lin)]=f_max-(f_max-f_lin)*x**10
    
    return f
    
#%% Initialization
files=np.array(sorted(glob.glob(source)))

#read all PPR, Dr
ppr_all=[]
dr_all=[]

for f in files:
    match = re.search(pattern, f)
    ppr_all = np.append(ppr_all,int(match.group(1))) 
    dr_all =  np.append(dr_all,int(match.group(2)))
    
#zeroing
Data=pd.read_excel(files[0])
snr=np.float64(Data['SNR [dB]'].values)
ppr=np.unique(ppr_all)
dr=np.unique(dr_all)
noise_avg=np.zeros((len(snr),len(ppr),len(dr)))
noise_low=np.zeros((len(snr),len(ppr),len(dr)))
noise_top=np.zeros((len(snr),len(ppr),len(dr)))
snr_hist=np.zeros((len(snr),len(ppr),len(dr)))
snr_avg=np.zeros((len(ppr),len(dr)))       

#graphics
colors = [cmap(i) for i in np.linspace(0, 1, len(np.unique(ppr_all)))]

#%% Main

#assemble data array
ctr=0  
for f in files:
    Data=pd.read_excel(f)
    i_ppr=np.where(ppr_all[ctr]==ppr)[0][0]
    i_dr=np.where(dr_all[ctr]==dr)[0][0]
    noise_avg[:,i_ppr,i_dr]=Data['Noise StDev [m/s]'].values
    noise_low[:,i_ppr,i_dr]=Data['Noise StDev (2.5% percentile) [m/s]'].values
    noise_top[:,i_ppr,i_dr]=Data['Noise StDev (97.5% percentile) [m/s]'].values
    snr_hist[:,i_ppr,i_dr]=Data['Occurrence'].values
    snr_avg[i_ppr,i_dr]=np.sum(Data['Occurrence'].values*snr)/np.sum(Data['Occurrence'].values)
    ctr+=1

Data=xr.Dataset()
Data['noise_avg']=xr.DataArray(noise_avg,coords={'snr':snr,'ppr':ppr,'dr':dr})
Data['noise_low']=xr.DataArray(noise_low,coords={'snr':snr,'ppr':ppr,'dr':dr})
Data['noise_top']=xr.DataArray(noise_top,coords={'snr':snr,'ppr':ppr,'dr':dr})
Data['snr_hist']=xr.DataArray(snr_hist,coords={'snr':snr,'ppr':ppr,'dr':dr})
Data['snr_avg']=xr.DataArray(snr_avg,coords={'ppr':ppr,'dr':dr})

#qc
Data['qc_noise']=(Data.noise_top-Data.noise_low<max_unc)*(Data.noise_top-Data.noise_low>0)*(Data.snr_hist>=min_N)

Data['noise_avg_qc']=Data['noise_avg'].where(Data.qc_noise)#.interpolate_na(dim='snr')
Data['noise_low_qc']=Data['noise_low'].where(Data.qc_noise)#.interpolate_na(dim='snr')
Data['noise_top_qc']=Data['noise_top'].where(Data.qc_noise)#.interpolate_na(dim='snr')


#modeling
snr_lin=np.zeros((len(ppr),len(dr)))
noise_lin=np.zeros((len(ppr),len(dr)))
noise_mod=np.zeros((len(snr_plot),len(ppr),len(dr)))
for i_dr in range(len(dr)):
    for i_ppr in range(len(ppr)):
        noise=Data.noise_avg_qc.isel(dr=i_dr,ppr=i_ppr).where(Data.snr<=snr_max)
        unc=(Data.noise_top_qc-Data.noise_low_qc).isel(dr=i_dr,ppr=i_ppr).where(Data.snr<=snr_max)

        try:
            params, _ = curve_fit(curve,snr[~np.isnan(noise)],np.log(noise[~np.isnan(noise)]),sigma=np.log(unc[~np.isnan(noise)]),
            p0=[-15,0.5],bounds=([-25,0.01],[-10,1]))
        except:
            params=[np.nan,np.nan,np.nan,np.nan]
        
        snr_lin[i_ppr,i_dr]=params[0]
        noise_lin[i_ppr,i_dr]=params[1]
        noise_mod[:,i_ppr,i_dr]=np.exp(curve(snr_plot,params[0],params[1]))
        
Data['snr_lin']=xr.DataArray(data=snr_lin,coords={'ppr':ppr,'dr':dr})
Data['noise_lin']=xr.DataArray(data=noise_lin,coords={'ppr':ppr,'dr':dr})
Data['noise_mod']=xr.DataArray(data=noise_mod,coords={'snr_plot':snr_plot,'ppr':ppr,'dr':dr})

Data['delta']=10**(Data.snr/10)
Data['noise_th']=(4*np.pi**0.5*f2**3/(Data.ppr*Data.dr/(c/(2*fs))*Data.delta**2)*(1+0.16*Data.delta/f2)**2)**0.5*(Nyquist*2)

#%% Plots
plt.close('all')

#all noise curves
plt.figure(figsize=(20,10))
snr_ref=-25
i_dr=0
for dri in dr:
    ax=plt.subplot(2,len(dr),i_dr+1)
    i_ppr=0
    for ppri in ppr:
        if i_dr==0:
            ax.semilogy(snr,Data.noise_avg_qc.sel(dr=dri,ppr=ppri),'.',color=colors[i_ppr],label=r'$N='+str(int(ppri))+'$')
            
        else:
            ax.semilogy(snr,Data.noise_avg_qc.sel(dr=dri,ppr=ppri),'.',color=colors[i_ppr])
        
        plt.errorbar(snr, Data.noise_avg_qc.sel(dr=dri,ppr=ppri),
                     yerr=[Data.noise_avg_qc.sel(dr=dri,ppr=ppri)-Data.noise_low_qc.sel(dr=dri,ppr=ppri),
                           Data.noise_top_qc.sel(dr=dri,ppr=ppri)-Data.noise_avg_qc.sel(dr=dri,ppr=ppri)],linestyle='none',color=colors[i_ppr],capsize=5)
        plt.plot(snr_plot,Data.noise_mod.sel(dr=dri,ppr=ppri),'-',color=colors[i_ppr])
        if plot_theory:
            plt.plot(snr,Data.noise_th.sel(dr=dri,ppr=ppri),'--',color=colors[i_ppr])
        
        i_ppr+=1
        
    plt.grid()
    plt.title(r'$M='+str(int(dri/(c/(2*fs))))+'$')
    plt.xlim([-30,snr_max])
    plt.ylim([0.01,30])
    
    if i_dr==0:
        plt.legend()
        plt.ylabel(r'$\sigma_T$ [m s$^{-1}$]')
    else:
        ax.set_yticklabels([])
    ax.set_xticklabels([])
    i_dr+=1
    
i_ppr=0
for ppri in ppr:
    ax=plt.subplot(2,len(ppr),i_ppr+1+len(ppr))
    i_dr=0
    for dri in dr:
        if i_ppr==0:
            ax.semilogy(snr,Data.noise_avg_qc.sel(ppr=ppri,dr=dri),'.',color=colors[i_dr],label=r'$M='+str(int(dri/(c/(2*fs))))+'$')
        else:
            ax.semilogy(snr,Data.noise_avg_qc.sel(ppr=ppri,dr=dri),'.',color=colors[i_dr])
            
        plt.errorbar(snr, Data.noise_avg_qc.sel(dr=dri,ppr=ppri),
                     yerr=[Data.noise_avg_qc.sel(dr=dri,ppr=ppri)-Data.noise_low_qc.sel(dr=dri,ppr=ppri),
                           Data.noise_top_qc.sel(dr=dri,ppr=ppri)-Data.noise_avg_qc.sel(dr=dri,ppr=ppri)],linestyle='none',color=colors[i_dr],capsize=5)
        plt.plot(snr_plot,Data.noise_mod.sel(dr=dri,ppr=ppri),'-',color=colors[i_dr])
        if plot_theory:
            plt.plot(snr,Data.noise_th.sel(dr=dri,ppr=ppri),'--',color=colors[i_dr])

        i_dr+=1
        
    plt.grid()
    plt.title(r'$N='+str(int(ppri))+'$')
    plt.xlim([-30,snr_max])
    plt.ylim([0.01,30])
    plt.xlabel('SNR [dB]')
    
    if i_ppr==0:
        plt.legend()
        plt.ylabel(r'$\sigma_T$ [m s$^{-1}$]')
    else:
        ax.set_yticklabels([])
    i_ppr+=1
plt.tight_layout()
  
#snr_lin and noise_lin
fig=plt.figure(figsize=(12,5))
ax = fig.add_subplot(1,2,1)
plt.pcolor(Data.snr_lin.T,cmap='RdYlGn_r')
for i_ppr in range(len(ppr)):
    for i_dr in range(len(dr)):
        ax.text(i_ppr+0.5, i_dr+0.5, f"{Data.snr_lin.isel(ppr=i_ppr,dr=i_dr):.1f}", 
                ha='center', va='center', color='k', fontsize=12,fontweight='bold')
plt.xticks(np.arange(len(ppr))+0.5,labels=[f'{int(ppri)}' for ppri in ppr])
plt.yticks(np.arange(len(dr))+0.5,labels=[f'{int(dri/(c/(2*fs)))}' for dri in dr])
plt.xlabel(r'$N$')
plt.ylabel(r'$M$')

ax = fig.add_subplot(1,2,2)
plt.pcolor(Data.noise_lin.T,cmap='RdYlGn_r')
for i_ppr in range(len(ppr)):
    for i_dr in range(len(dr)):
        ax.text(i_ppr+0.5, i_dr+0.5, f"{Data.noise_lin.isel(ppr=i_ppr,dr=i_dr):.3f}", 
                ha='center', va='center', color='k', fontsize=12,fontweight='bold')
plt.xticks(np.arange(len(ppr))+0.5,labels=[f'{int(ppri)}' for ppri in ppr])
plt.yticks(np.arange(len(dr))+0.5,labels=[])
plt.xlabel(r'$N$')
plt.tight_layout()