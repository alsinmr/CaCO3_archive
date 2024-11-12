#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:37:24 2024

@author: albertsmith

This file creates all the simulation results found in the folder T1p_3spin_run0
"""

import numpy as np
import sys
sys.path.append('/Users/albertsmith/Documents/GitHub/')  # Path to SLEEPY folder
import SLEEPY as sl
sl.Defaults['verbose']=False
sl.Defaults['parallel']=True #Maybe it makes sense to change this on the server?
import os
from time import time


from NMRparameters import deltaCSA,etaCSA,alphaCSA,betaCSA,gammaCSA   #13C CSA
from NMRparameters import delta,beta,gamma                      #H-C dipole 
from NMRparameters import deltaHCSA,alphaHCSA,betaHCSA,gammaHCSA,HCS  #1H CS/CSA
from NMRparameters import deltaHH,betaHH,gammaHH              #H–H dipole
    
    
    
#%% Build 3-spin simulations
ex=sl.ExpSys(v0H=400,Nucs=['13C','1H','1H'],vr=5000,pwdavg=sl.PowderAvg(q=3))
ex.set_inter('CSA',i=0,delta=deltaCSA,eta=etaCSA,euler=[alphaCSA,betaCSA,gammaCSA])
for k in range(2):
    ex.set_inter('dipole',i0=0,i1=k+1,delta=delta[k],euler=[0,beta[k],gamma[k]])
    ex.set_inter('CSA',i=k+1,delta=deltaHCSA[k],euler=[alphaHCSA[k],betaHCSA[k],gammaHCSA[k]])
    ex.set_inter('CS',i=k+1,ppm=HCS[k])
ex.set_inter('dipole',i0=1,i1=2,delta=deltaHH[0,1],euler=[0,betaHH[0,1],gammaHH[0,1]])


#%% Sweep over the conditions
ntc=31
nSD=11

tc0=np.logspace(-6,-3,ntc)
kSD0=np.logspace(0,2,nSD)

counter=0
folder='T1p_3spin_run0'


nt=500
t0=time()

for p,tc in enumerate(tc0):
    for q,kSD in enumerate(kSD0):
        file=os.path.join(folder,f'R1p_{-np.log10(tc):.1f}_{np.log10(kSD):.1f}').replace('.','p')+'.npy'
        
        if os.path.exists(file):continue #Already calculated
        
        I=np.zeros([5,nt])
        
        L=ex.Liouvillian()
        L.add_SpinEx([1,2],tc)
        
        for i in range(1,3):
            L.add_relax(Type='SpinDiffusion',i=i,k=kSD)
            
        
        for k,v1 in enumerate([2000,7000,12000,14000,22000]):
            try:
                seq=L.Sequence().add_channel('13C',v1=v1)
        
                rho=sl.Rho(rho0='13Cx',detect='13Cx')
                rho.DetProp(seq,n=nt)
                I[k]=rho.I[0].real
            except:
                print(file) 
                
        np.save(file,I,allow_pickle=False)
        elapsed=(time()-t0)/60
        counter+=1
        total_time=elapsed*ntc*nSD/counter
        print(f'{elapsed:.2f} minutes out of ~{total_time:.2f} minutes')