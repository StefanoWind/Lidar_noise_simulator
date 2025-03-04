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
matplotlib.rcParams['font.size'] = 14

#%% Inputs
M=32
delta=0.1
f1=1/M*3
f2=0.02

#%% Initialization
f=np.arange(M+1)/M-0.5#normalized frequencies

#%% Main
phi=delta/(2*np.pi)**0.5/f2*np.exp(-(f-f1)**2/(2*f2**2))+1#mean spectrum

#%% Plots
plt.figure(figsize=(16,5))
plt.bar(f,phi,color=(0,0,0,0.25),width=0.005)
plt.bar(f,f**0,color='k',width=0.005)

plt.plot(f,phi,'.k',markersize=15)
plt.xlabel('$f/f_s$ ')
plt.ylabel('$S$')
plt.grid()
plt.xticks([-0.5,-0.25,0,0.25,0.5])
plt.yticks([])

plt.tight_layout()