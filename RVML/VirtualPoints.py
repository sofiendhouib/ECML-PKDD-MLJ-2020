#!/usr/bin/env python

# Copyright 2015 Michaël Perrot

# This file is part of RVML.

# RVML is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# FoobarRVML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with RVML.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import transport
from scipy.spatial.distance import cdist

def classBasedVP(X,Y):
    uni,inverse = np.unique(Y,return_inverse=True)
    identity = np.identity(len(uni))
    VirtualPoints = np.array([identity[i,:] for i in inverse])
    return VirtualPoints

def transportBasedVP(X,Y,reg=0.5,eta=10):
    d = X.shape[1] 
    nSource = X.shape[0]
    # The threshold is used to stop the selection of landmarks
    threshold = np.mean(cdist(X,X,metric='euclidean'))
    idxSource = np.arange(nSource)
    # Selection of the first landmark
    idxTarget = [idxSource[np.argmax(cdist(np.zeros((1,d)),X[idxSource],metric='euclidean'))]]
    idxSource = idxSource[idxSource != idxTarget[-1]]
    distances = cdist(X[idxTarget],X[idxSource],metric='euclidean')
    # We want at least one example per class and the nearest landmark of each point should be at a distance lower than the threshold
    while len(idxTarget) < len(np.unique(Y)) or np.amax(np.amin(distances,axis=0)) > threshold:
        distances = np.sum(distances,axis=0)/len(idxTarget)
        idxTarget.append(idxSource[np.argmax(distances)])
        idxSource = idxSource[idxSource != idxTarget[-1]]
        distances = cdist(X[idxTarget],X[idxSource],metric='euclidean')

    # Optimal Transport using the toolbox provided by Remi Flamary for the paper
    # Optimal Transport for Domain Adaptation
    # N. Courty, R. Flamary, D. Tuia, and A. Rakotomamonjy
    Source = X
    Target = X[idxTarget]
    nTarget = Target.shape[0]
    
    wSource = np.array([1./nSource]*nSource)
    wTarget = np.array([1./nTarget]*nTarget)
    
    distances = cdist(Source,Target,metric='euclidean')
    
    transp1 = transport.computeTransportSinkhornLabelsLpL1(wSource,Y,wTarget,distances,reg,eta)
    transp1 = np.dot(np.diag(1/np.sum(transp1,1)),transp1) 

    VirtualPoints = np.dot(transp1,Target)
    return VirtualPoints
