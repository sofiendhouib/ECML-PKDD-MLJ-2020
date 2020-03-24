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

import os, sys, traceback
import numpy as np
from RVML import *
from VirtualPoints import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import rbf_kernel

np.set_printoptions(threshold=np.nan) # To print everything with NumPy

# Dataset name
dataset = "splice"

#  NN-Classifier Parameter
k = 1

# Training data
dataTrain = np.loadtxt("data/"+dataset+".train",delimiter=",")
XTrain = dataTrain[:,1:]
YTrain = dataTrain[:,0]

# Normalization
XMean = np.mean(XTrain,axis=0)
XStd = np.std(XTrain,axis=0)
XStd = np.where(XStd == 0,1,XStd) # In case of a constant attribute
XTrain = (XTrain - XMean)/(3*XStd)

# Kernel and learner
kernel_args = {"gamma": 1/(2*(np.mean(cdist(XTrain,XTrain,'euclidean'))**2))}
kernel_fct = lambda A,B : rbf_kernel(A,B,**kernel_args)

#learner = linearRVML
learner = lambda X,V,l : kernelizedRVML(X,V,l,kernel_fct)
#VP = transportBasedVP
VP = transportBasedVP

# Learning the metric with a 5-folds cross-validation
learnedMetric = learner(XTrain,
                 VP(XTrain,YTrain),
                 crossValidateLambda(learner,
                                     VP,
                                     KNeighborsClassifier(n_neighbors=k,metric='euclidean'),
                                     XTrain,
                                     YTrain))

# Testing data
dataTest = np.loadtxt("data/"+dataset+".test",delimiter=",")
XTest = dataTest[:,1:]
YTest = dataTest[:,0]

# Normalization
XTest = (XTest - XMean)/(3*XStd)

# Applying the metric and learning a classifier
clf = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
print("Accuracy on "+dataset+":",100*clf.fit(learnedMetric(XTrain),YTrain).score(learnedMetric(XTest),YTest))