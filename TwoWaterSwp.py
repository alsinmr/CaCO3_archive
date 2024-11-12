#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:19:39 2024

@author: albertsmith

This file creates all the simulation results found in the folder T1p_5spin_run0
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

#%% Build 5-spin simulations
ex=sl.ExpSys(v0H=400,Nucs=['13C','1H','1H','1H','1H'],vr=5000,pwdavg=sl.PowderAvg(q=2))
ex.set_inter('CSA',i=0,delta=deltaCSA,eta=etaCSA,euler=[alphaCSA,betaCSA,gammaCSA])
for k in range(4):
    ex.set_inter('dipole',i0=0,i1=k+1,delta=delta[k],euler=[0,beta[k],gamma[k]])
    ex.set_inter('CSA',i=k+1,delta=deltaHCSA[k],euler=[alphaHCSA[k],betaHCSA[k],gammaHCSA[k]])
    ex.set_inter('CS',i=k+1,ppm=HCS[k])
    for m in range(k+1,4):
        ex.set_inter('dipole',i0=k+1,i1=m+1,delta=deltaHH[k,m],euler=[0,betaHH[k,m],gammaHH[k,m]])


#%% Sweep over the conditions
ntc=81
nSD=12

tc0=np.logspace(-6,-2,ntc)
kSD0=np.concatenate([[0],np.logspace(0,2,nSD-1)])

counter=0
folder=f'T1p_5spin_run{counter}'
while os.path.exists(folder):
    if not(os.path.exists(os.path.join(folder,'tc.npy'))):break     #Parameters not saved in folder
    if not(os.path.exists(os.path.join(folder,'kSD.npy'))):break    #Parameters not saved in folder
    #Note, if somehow the parameter files get deleted, but not the runs, this will completely break ;-)
    
    #Check if existing parameters match the desired ones
    if np.all(tc0==np.load(os.path.join(folder,'tc.npy'),allow_pickle=False)) and \
        np.all(kSD0==np.load(os.path.join(folder,'kSD.npy'),allow_pickle=False)):
            break
        
    #Otherwise, make a new folder    
    counter+=1
    folder=f'T1p_5spin_run{counter}'
    
if not(os.path.exists(folder)):os.mkdir(folder)    
if not(os.path.exists(os.path.join(folder,'tc.npy'))):
    np.save(os.path.join(folder,'tc.npy'),tc0)
if not(os.path.exists(os.path.join(folder,'kSD.npy'))):
    np.save(os.path.join(folder,'kSD.npy'),kSD0)
            


q=int(sys.argv[1])
kSD=kSD0[q]

nt=500
t0=time()
counter=0
for k,tc in enumerate(tc0):
    file=os.path.join(folder,f'tc{k:03d}_kSD{q:02d}.npy')
    if os.path.exists(file):continue
    
    I=np.zeros([5,nt])
    L=ex.Liouvillian()
    L.add_SpinEx([1,2], tc)
    L.add_SpinEx([3,4], tc)
    for i in range(1,5):
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
    elapsed=(time()-t0)/3600
    counter+=1
    total_time=elapsed*ntc/counter
    print(f'{elapsed:.2f} hours elapsed out of ~{total_time:.2f} hours')
        
    

