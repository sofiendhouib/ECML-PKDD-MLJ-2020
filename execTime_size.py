#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import path as pathSys
pathSys.append('../')

import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics 
from sklearn import model_selection
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.datasets import make_blobs

from sklearn.pipeline import Pipeline
import similarity_learning as sl
import myIOFuncs as ioFuncs
from time import time




#%%
ridgeSimLearner = sl.bilinearSimilarityLearner(algorithm= 'closed-form')
sizes = [100*i for i in range(1,31, 6)]
dims = [10*i for i in range(1,21, 4)]
nbReps = 100
execTimes = np.zeros((len(sizes), len(dims), nbReps))
for i, size in enumerate(sizes):
    print(i)
    for j, dim in enumerate(dims):
#        print(j)
        X, y = make_blobs(n_samples= size, n_features= dim, centers= 2)
        for k in range(nbReps):
            startTime = time()
            ridgeSimLearner.fit(X,y)
            endTime = time()
            execTimes[i,j,k] = endTime - startTime
            
np.save("time_size_dim.npy", execTimes)
#%%
plt.close('all')
#plt.figure()
#plt.plot(sizes, execTimes.mean(axis= -1))

execTimes= np.load("time_size_dim.npy")
plt.figure()
for i, _ in enumerate(dims):
    ioFuncs.plotWithStd(sizes, execTimes[:,i,:], axis= -1)
plt.legend(["d= %d"%d for d in dims])
plt.xlabel("size m")
plt.ylabel("exeution time (s)")
plt.savefig("timeVSdim.png")

#plt.figure()
#plt.plot(dims, execTimes.mean(axis= -1).T)

plt.figure()
for i, _ in enumerate(sizes):
    ioFuncs.plotWithStd(dims, execTimes[i,:,:], axis= -1)
plt.legend(["m= %d"%m for m in sizes])
plt.xlabel("dimesnion d")
plt.ylabel("exeution time (s)")
plt.savefig("timeVSsize.png")