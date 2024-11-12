#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:40:09 2024

@author: albertsmith
"""

import numpy as np
import SLEEPY as sl
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

#%% Load the experimental data
nexp=11

texp=np.zeros([nexp])
files=['T1rho_13C_100pdisACC_10kHz/T1rho_13C_100pdisACC_10khz_spinlock_strength_{}kHz.txt',
       'T1rho_13C_100pdisACC_10kHz/T1rho_13C_100pdisACC_10khz_spinlock_strength_{}kHz_repeat.txt']
exp_data={}
for q,file in enumerate(files):
    Iexp=np.zeros([5,nexp])
    for m,v0 in enumerate([2,7,12,14,22]):
        with open(file.format(v0),'r') as f:
            f.readline()
            for k,line in enumerate(f):
                texp[k],Iexp[m,k]=line.strip().split()
            Iexp[m]/=Iexp[m,3]
        
    exp_data.update({'t':texp,f'I{q}':Iexp,'v1':[2,7,12,14,22]})
    
    
#%% Load simulation data
#(currently, may not be sorted if multiple folders populated- probably not important)


kSD=[]
tc=[]
I=[]
folder='T1p_5spin_run0'

tc0=np.load(os.path.join(folder,'tc.npy'),allow_pickle=False)
kSD0=np.load(os.path.join(folder,'kSD.npy'),allow_pickle=False)

skip=0
tsim=np.arange(500)*.0002
iexp=np.array([np.argmin(np.abs(tsim-t)) for t in texp]) #Index for comparing simulation to experiment

for file in sorted(os.listdir(folder)):
    if file[:2]=='tc' and file[6:9]=='kSD':
        I.append([])
        tc.append(tc0[int(file[2:5])])
        kSD.append(kSD0[int(file[9:11])])
        I0=np.load(os.path.join(folder,file),allow_pickle=False)
        for k in range(5):
            scale=((Iexp[k,skip:]*Iexp[k,skip:]).sum())**(-1)*(I0[k,iexp[skip:]]*Iexp[k,skip:]).sum()
        
            I[-1].append(I0[k]/scale)
    
kSD=np.array(kSD)
tc=np.array(tc)
I=np.array(I)

sim_data={'t':np.arange(500)*.0002,'tc':tc,'kSD':kSD,'I':I}



nsims=len(kSD)




#%% plotting function for comparing sims
v1=[2,7,12,14,22]


def get_tc_kSD(tc=None,kSD=None):
    if kSD is not None:
        kSD=sim_data['kSD'][np.argmin(np.abs(kSD-sim_data['kSD']))]
        i0=kSD==sim_data['kSD']
    else:
        i0=np.ones(sim_data['kSD'].size,dtype=bool)
        
    if tc is not None:
        tc=sim_data['tc'][np.argmin(np.abs(tc-sim_data['tc']))]
        i1=tc==sim_data['tc']
    else:
        i1=np.ones(sim_data['tc'].size,dtype=bool)
    i0=np.logical_and(i0,i1)
    
    return {key:value[i0] if len(value)==len(sim_data['tc']) else value for key,value in sim_data.items()}
    
def get_error(skip:int=0,q=None,kSD=None,tc=None):
    
    sim_data=get_tc_kSD(tc=tc,kSD=kSD)
    nsims=len(sim_data['tc'])
    
    error=[]
    q=np.arange(5) if q is None else np.atleast_1d(q)
    for k,Ie in enumerate(Iexp):
        if k not in q:continue
        I0=sim_data['I'][:,k,iexp].reshape([nsims,nexp])
        error.append(((Ie-I0)**2).sum(1))
    error=np.array(error).sum(0)
    
    error[np.isnan(error)]=error[np.logical_not(np.isnan(error))].max()

    return error

def plot_fixed_kSD(i:int=None,ax:list=None,skip:int=0,kSD:float=None,tc:float=None,q=None):


    sim_data=get_tc_kSD(tc=tc,kSD=kSD)
    
    
    if i is None:
        error=get_error(skip=skip,kSD=kSD,tc=tc) if q is None else get_error(skip,q=q,tc=tc,kSD=kSD)
        i=np.argmin(error)
        
        
    if ax is None:
        fig=plt.figure()
        ax=[fig.add_subplot(2,3,k+1) for k in range(6)]
    else:
        assert len(ax)==6,"Must provide 6-element list/array of axes"
        fig=ax[0].figure
        for a in ax:a.cla()
        
    skip=0    
    for k,a in enumerate(ax[:-1]):
        sc=exp_data['I0'][k].max()
        a.plot(sim_data['t']*1e3,sim_data['I'][i,k]/sc,color='red')
        
        a.scatter(texp*1e3,exp_data['I0'][k]/sc,marker='o',color='black')
        a.set_ylim([0,1])
        a.set_xlabel('t / ms')
        
        a.text(50,.8,f'{v1[k]} kHz')
        
    ax[-1].plot(sim_data['t']*1e3,sim_data['I'][i,:].T)
    ax[-1].set_xlabel('t / ms')
    ax[-1].legend([f'{k} kHz' for k in v1])
    ax[-1].set_ylim([0,1])
        
    ax[1].set_title(rf'$k_{{SD}}$ = {sim_data["kSD"][i]:.1f} s$^{{-1}}$, $\tau_c$ = {sim_data["tc"][i]*1e6:.1f} $\mu s$')
    for a in ax:
        if not(a.is_first_col()):a.set_yticklabels([])
        if not(a.is_last_row()):a.set_xticklabels([])
    fig.set_size_inches([8,5])
    fig.tight_layout()
    return fig
    
    
def plot_variable_kSD():
    best=[]
    error=[]
    tc0=get_tc_kSD(kSD=0)['tc']
    kSD0=get_tc_kSD(tc=1000)['kSD']
    for tc in tc0:
        error.append(0)
        best.append([])
        for q in range(5):
            e=get_error(tc=tc,q=q)
            best[-1].append(np.argmin(e))
            error[-1]+=e[best[-1][-1]]
    
    error=np.array(error)
    best=np.array(best)        
    
    i=np.argmin(error)
    best=best[i]
    
    tc=tc0[i]
    # fig=plt.figure()
    # ax=[fig.add_subplot(2,3,k+1) for k in range(6)]
    fig,ax=plt.subplots(2,3,sharex=False)
    ax=ax.flatten()
    
    sim_data=get_tc_kSD(tc=tc)
    v1=[2,7,12,14,22]
    for k,a in enumerate(ax[:-1]):
        sc=exp_data['I0'][k].max()
        a.plot(sim_data['t']*1e3,sim_data['I'][best[k]][k]/sc,color='red')
        a.scatter(exp_data['t']*1e3,exp_data['I0'][k]/sc,color='black',marker='o')
        a.set_ylim([0,1.05])
        if not(a.is_last_row()):a.set_xticklabels([])
        if not(a.is_first_col()):a.set_yticklabels([])
        if a.is_last_row():a.set_xlabel('t / ms')
        a.text(25,.8,f'{v1[k]} kHz'+'\n'+fr'$k_{{SD}}$ = {kSD0[best[k]]:.1f} s$^{{-1}}$',verticalalignment='top')
        ax[-1].plot(sim_data['t']*1e3,sim_data['I'][best[k]][k])
    ax[-1].legend([f'{v10} kHz' for v10 in v1])
    ax[-1].set_xlabel('t / ms')
        
    ax[1].set_title(fr'$\tau_c$ = {tc*1e6:.1f} $\mu$s')
    fig.set_size_inches([8,5])
    fig.tight_layout()
    return fig