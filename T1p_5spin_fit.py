#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:14:05 2024

@author: albertsmith
"""

import os
import numpy as np
import matplotlib.pyplot as plt


#%% Load all data from folders 
#(currently, may not be sorted if multiple folders populated- probably not important)

k=1
kSD=[]
tc=[]
I=[]
folder=f'T1p_5spin_run{k}'
while os.path.exists(folder):
    tc0=np.load(os.path.join(folder,'tc.npy'),allow_pickle=False)
    kSD0=np.load(os.path.join(folder,'kSD.npy'),allow_pickle=False)
    
    for file in sorted(os.listdir(folder)):
        if file[:2]=='tc' and file[6:9]=='kSD':
            tc.append(tc0[int(file[2:5])])
            kSD.append(kSD0[int(file[9:11])])
            I.append(np.load(os.path.join(folder,file),allow_pickle=False))
    k+=1
    folder=f'T1p_5spin_run{k}'
    
kSD=np.array(kSD)
tc=np.array(tc)
I=np.array(I)

exp_data={'tc':tc,'kSD':kSD,'I':I}

nsims=len(kSD)


#%% Load experimental data
nexp=11
Iexp=np.zeros([5,nexp])
texp=np.zeros([nexp])
file='T1rho_13C_100pdisACC_10kHz/T1rho_13C_100pdisACC_10khz_spinlock_strength_{}kHz.txt'
for m,v0 in enumerate([2,7,12,14,22]):
    with open(file.format(v0),'r') as f:
        f.readline()
        for k,line in enumerate(f):
            texp[k],Iexp[m,k]=line.strip().split()
        Iexp[m]/=Iexp[m,3]
        
# ax=plt.subplots()[1]
# ax.plot(texp*1e3,Iexp.T)
# ax.set_xlabel('t / ms')
# ax.legend(('2 kHz','7 kHz','12 kHz','14 kHz','22 kHz'))

#%% Find the best fitting data
# Here, we first pick out the simulated time points to compare to experiment
# Is this the best idea? For smooth parts of the curve,should be fine
# For oscillatory regions, probably oscillations aren't exact in case the
# dipole couplings aren't exactly right....

tsim=np.arange(500)*.0002
iexp=np.array([np.argmin(np.abs(tsim-t)) for t in texp]) #Index for comparing simulation to experiment

skip=0
error=[]
for k,Ie in enumerate(Iexp):
    I0=I[:,k,iexp].reshape([nsims,nexp])
    # I0[I0==0]=1e-3
    scale=((Ie[skip:]*Ie[skip:]).sum(-1))**(-1)*(I0[:,skip:]*Ie[skip:]).sum(-1)
    # scale=I0[:,skip]/Ie[skip]
    
    # scale=(Ie/I0).mean(1)
    error.append(((Ie-(I0.T/scale).T)**2).sum(1))
error=np.array(error).sum(0)

error[np.isnan(error)]=error[np.logical_not(np.isnan(error))].max()

    

#%% Function for plotting experiment vs. sim
# fig=plt.figure()
# ax=[fig.add_subplot(2,3,k+1) for k in range(6)]

v1=[2,7,12,14,22]

def exp_v_sim(i:int=None,ax:list=None):
    if i is None:i=np.argmin(error)
    if ax is None:
        fig=plt.figure()
        ax=[fig.add_subplot(2,3,k+1) for k in range(6)]
    else:
        assert len(ax)==6,"Must provide 6-element list/array of axes"
        fig=ax[0].figure
        for a in ax:a.cla()
        
        
    for k,a in enumerate(ax[:-1]):
        a.plot(tsim*1e3,I[i,k],color='red')
        
        scale=((Iexp[k,skip:]*Iexp[k,skip:]).sum())**(-1)*(I[i,k,iexp[skip:]]*Iexp[k,skip:]).sum()
        # scale=I[i,k,skip]/Iexp[k,skip]
        
        a.scatter(texp*1e3,Iexp[k]*scale,marker='o',color='black')
        a.set_ylim([0,1])
        a.set_xlabel('t / ms')
        
        a.text(50,.8,f'{v1[k]} kHz')
        
    ax[-1].plot(tsim*1e3,I[i,:].T)
    ax[-1].set_xlabel('t / ms')
    ax[-1].legend([f'{k} kHz' for k in v1])
    ax[-1].set_ylim([0,1])
        
    ax[1].set_title(rf'$k_{{SD}}$ = {kSD[i]:.0f} s$^{{-1}}$, $\tau_c$ = {tc[i]*1e6:.1f} $\mu s$')
    fig.set_size_inches([10,5.75])
    fig.tight_layout()