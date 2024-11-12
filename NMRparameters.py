#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 14:05:08 2024

@author: albertsmith

This file calculates or loads all the NMR parameters from the mhc.magres file
"""

import numpy as np
import SLEEPY as sl

#%% Load positions, CSA, box size
pos=[]
nuc=[]
index=[]
CSA=[]
with open('mhc.magres','r') as f:
    for line in f:
        if line[:4]=='atom':
            nuc.append(line.split()[2])
            index.append(int(line.split()[3])-1)
            pos.append([float(l) for l in line.split()[4:7]])
        elif line[:7]=='lattice':
            box=np.diag(np.array([float(l) for l in line.split()[1:10]]).reshape([3,3]))
        elif line[:2]=='ms':
            CSA.append(np.array([float(l) for l in line.split()[3:12]]).reshape([3,3]))
        
            
nuc=np.array(nuc)
index=np.array(index)
pos=np.array(pos)
CSA=np.array(CSA)

#%% Nearest protons to Carbon #1
iC=8

i=np.argmax(np.logical_and(nuc=='C',index==iC))
posC=pos[i]
posH0=pos[nuc=='H']
Dpos=posH0-posC

# Periodic boundary correction
i=Dpos>box/2
ib=np.argwhere(i).T[1]
Dpos[i]=Dpos[i]-box[ib]

i=Dpos<-box/2
ib=np.argwhere(i).T[1]
Dpos[i]=Dpos[i]+box[ib]

dHC=np.sqrt(((Dpos)**2).sum(-1))
iHC=np.argsort(dHC)
iH0=iHC[:2]
iH1=np.zeros(2,dtype=int)

# Where are the partners of the nearest two waters?
for k,posH in enumerate(posH0[iH0]):
    DposHH=posH0-posH

    # Periodic boundary correction
    i=DposHH>box/2
    ib=np.argwhere(i).T[1]
    DposHH[i]=DposHH[i]-box[ib]

    i=DposHH<-box/2
    ib=np.argwhere(i).T[1]
    DposHH[i]=DposHH[i]+box[ib]
    
    dHH=np.sqrt(((DposHH)**2).sum(-1))
    
    iH1[k]=np.argsort(dHH)[1]
iH=[iH0[0],iH1[0],iH0[1],iH1[1]]

#%% HC Dipole coupling calculations
delta=sl.Tools.dipole_coupling(dHC[iH]/10,'1H','13C')

v=np.array([Dpos[i0]/np.sqrt((Dpos[iHC[i0]]**2).sum(-1)) for i0 in iH])
beta=np.arccos(v[:,2])
gamma=np.arctan2(v[:,1],v[:,0])


#%% H–H dipole coupling calculations
DposHH=[]
for i00 in iH:
    for i01 in iH:
        DposHH.append(posH0[i00]-posH0[i01])
DposHH=np.array(DposHH)


DposHH=[]
for i00 in iH:
    for i01 in iH:
        DposHH.append(posH0[i00]-posH0[i01])
DposHH=np.array(DposHH)

i=DposHH>box/2
ib=np.argwhere(i).T[1]
DposHH[i]=DposHH[i]-box[ib]

i=DposHH<-box/2
ib=np.argwhere(i).T[1]
DposHH[i]=DposHH[i]+box[ib]

dHH=np.sqrt(((DposHH)**2).sum(-1))
dHH[[0,5,10,15]]=1
deltaHH=sl.Tools.dipole_coupling(dHH/10,'1H','1H').reshape([4,4])


v=(DposHH.T/dHH).T
betaHH=np.arccos(v[:,2]).reshape([4,4])
gammaHH=np.arctan2(v[:,1],v[:,0]).reshape([4,4])

#%% 13C CSA

csa=CSA[np.logical_and(nuc=='H',index==iC)].squeeze()

D,V=np.linalg.eigh(csa)   #Get eigenvalues, eigenvectors 
i=np.argsort(np.abs(D))
D,V=D[i[[1,0,2]]],V[:,i[[1,0,2]]]     #Ordering is |azz|>=|axx|>=|ayy|
V=V*np.sign(np.linalg.det(V))
D-=D.mean()  #Discard the isotropic chemical shift

deltaCSA=D[2]
etaCSA=(D[1]-D[0])/D[2]

betaCSA=np.arccos(V[2,2])
alphaCSA=np.arctan2(V[2,1],V[2,0])
gammaCSA=np.arctan2(V[1,2],V[0,2])

#%% 1H CS/CSA
alphaHCSA,betaHCSA,gammaHCSA,deltaHCSA,etaHCSA,HCS=[],[],[],[],[],[]
for q in iH:
    csa=CSA[np.logical_and(nuc=='H',index==q)].squeeze()

    D,V=np.linalg.eigh(csa)   #Get eigenvalues, eigenvectors 
    i=np.argsort(np.abs(D))
    D,V=D[i[[1,0,2]]],V[:,i[[1,0,2]]]     #Ordering is |azz|>=|axx|>=|ayy|
    V=V*np.sign(np.linalg.det(V))
    HCS.append(D.mean())
    D-=D.mean()  #Remove the isotropic chemical shift

    deltaHCSA.append(D[2])
    etaHCSA.append((D[1]-D[0])/D[2])

    betaHCSA.append(np.arccos(V[2,2]))
    alphaHCSA.append(np.arctan2(V[2,1],V[2,0]))
    gammaHCSA.append(np.arctan2(V[1,2],V[0,2]))