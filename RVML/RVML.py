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

import random
import numpy as np
from numpy.linalg import inv

def linearRVML(X,V,l):
    """
    X is a n x d matrix where n is the number of examples and d is the input dimension
    V is a n x d' matrix where n is the number of examples and d' is the ouput dimension
    l is the regularision parameter

    Return a function used to project an array of examples in the learned space
    """
    n = X.shape[0]
    d = X.shape[1]
    if d <= n:
        L = inv(X.T.dot(X) + l*n*np.eye(d)).dot(X.T).dot(V)
    else:
        L = X.T.dot(inv(X.dot(X.T) + l*n*np.identity(n))).dot(V)
    return lambda ToProject : ToProject.dot(L)

def kernelizedRVML(X,V,l,kernel_fct):
    """
    X is a n x d matrix where n is the number of examples and d is the input dimension
    V is a n x d' matrix where n is the number of examples and d' is the ouput dimension
    l is the regularision parameter
    kernel_fct is a function used to compute the dot product between arrays of examples, e.g. a call to kernel_fct(X,X) should return a n x n matrix

    Return a function used to project an array of examples in the learned space
    """
    n = X.shape[0]
    KL = inv(kernel_fct(X,X)+l*n*np.identity(n)).dot(V)
    return lambda ToProject : kernel_fct(ToProject,X).dot(KL)

##################
# MISC FUNCTIONS #
##################

def crossValidateLambda(fctRVML,fctVP,clf,X,Y,nbFolds=5):
    lRange = [10**p for p in range(-5,2)]
    folds = foldsSplit(Y,nbFolds)
    scores = np.zeros(len(lRange))
    for j in range(nbFolds):
        XTrain = np.delete(X,folds[j],axis=0)
        YTrain = np.delete(Y,folds[j],axis=0)
        
        XTest = X[folds[j]]
        YTest = Y[folds[j]]

        metrics = [fctRVML(XTrain,fctVP(XTrain,YTrain),l) for l in lRange]
            
        scores = np.add(scores,np.array([clf.fit(metric(XTrain),YTrain).score(metric(XTest),YTest) for metric in metrics]))
    scores = scores/nbFolds
    return lRange[np.argmax(scores)]

def foldsSplit(Y,nbFolds=5):
    """
    Given the labels of the examples, returns k folds with approximately the sam
e number of examples in each fold for each class.
    """
    folds = [[] for j in range(nbFolds)]
    for c in np.unique(Y):
        idx = [i for (i, val) in enumerate(Y) if val == c]
        assert(len(idx)>=nbFolds) # At least one example of each class in each fold
        random.shuffle(idx)
        for j in range(len(idx)):
            folds[j%nbFolds].append(idx[j])
    return folds
