#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:09:59 2024

@author: albertsmith
"""

import numpy as np
import matplotlib.pyplot as plt

#%% Load experiments
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

#%% Load Simulations
nt=500
ntc=31
nSD=11
tc0=np.logspace(-6,-3,ntc)
kSD0=np.logspace(0,2,nSD)

I=np.zeros([tc0.size,kSD0.size,5,nt])
for p,tc in enumerate(tc0):
    for q,kSD in enumerate(kSD0):
        file=f'T1p_3spin_run0/R1p_{-np.log10(tc):.1f}_{np.log10(kSD):.1f}'.replace('.','p')+'.npy'
        I[p,q]=np.load(file,allow_pickle=False)


#%% Which time points match experimental time points?
tsim=np.arange(500)*.0002
iexp=np.array([np.argmin(np.abs(tsim-t)) for t in texp]) #Index for comparing simulation to experiment


skip=0
error=[]
for k,Ie in enumerate(Iexp):
    I0=I[:,:,k,iexp[skip:]].reshape([31*11,nexp-skip])
    I0[I0==0]=1e-3
    scale=(Ie[skip:]/I0).mean(1)
#     scale=Ie[skip:][0]
    error.append(((Ie[skip:]-(scale*I0.T).T)**2).sum(1))
error=np.array(error).sum(0)

i=np.argmin(error)

iSD=np.mod(i,nSD)
kSDf=kSD0[iSD]
i-=iSD
itc=i//nSD
# itc=np.argmin(np.abs(tc0-50e-6))
tcf=tc0[itc]

v1=[2,7,12,14,22]
def plot_fixed_kSD_3spin():
    fig=plt.figure()
    ax=[fig.add_subplot(2,3,k+1) for k in range(6)]
    
    
    for k,a in enumerate(ax[:-1]):
        I0=I[itc,iSD,k,iexp[skip:]]
        scale=(Iexp[k,skip:]/I0.T).mean()
    #     scale=Iexp[k][0]
        a.scatter(texp[:]*1000,Iexp[k][:],color='black',s=10)
        a.plot(tsim*1000,I[itc,iSD,k]*scale,color='red')
        a.text(50,.8,f'{v1[k]} kHz')
        a.set_ylim([0,1])
        a.set_xlabel('t / ms')
        ax[-1].plot(tsim*1000,I[itc,iSD,k]*scale,label=fr'$\nu_1$ = {v1[k]} kHz')
    ax[-1].legend()
    ax[-1].set_xlabel('t / ms')
    ax[1].set_title(rf'$k_{{SD}}$ = {kSD:.0f} s$^{{-1}}$, $\tau_c$ = {tcf*1e6:.1f} $\mu s$')
    fig.set_size_inches([8,5])
    fig.tight_layout()
    return fig