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


plotOnly = True


#%%
sizes = [1000*i for i in range(1,11)]
dims = [1000*i for i in range(1,11)]
if not plotOnly:
    ridgeSimLearner = sl.bilinearSimilarityLearner(algorithm= 'closed-form')
    # sizes = [int(100*i) for i in np.linspace(1,200,10)]
    sizes = [1000*i for i in range(1,11)]
    
    dims = [100*i for i in range(1,11)]
    # dims = [170]
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
else:
    plt.close('all')
    # plt.figure()
    # plt.plot(sizes, execTimes.mean(axis= -1))
    
    execTimes= np.load("time_size_dim.npy")
    plt.figure()
    for i, _ in enumerate(dims):
        ioFuncs.plotWithConf(sizes, execTimes[:,i,:], axis= -1)
    plt.legend(["d= %d"%d for d in dims])
    plt.xlabel("size m")
    plt.ylabel("execution time (s)")
    plt.savefig("timeVSsize.png")
    
    # plt.figure()
    # plt.plot(dims, execTimes.mean(axis= -1).T)
    
    plt.figure()
    for i, _ in enumerate(sizes):
        ioFuncs.plotWithConf(dims, np.sqrt(execTimes[i,:,:]), axis= -1)
    plt.legend(["m= %d"%m for m in sizes])
    plt.xlabel("dimension $d$")
    # plt.xticks([d**2 for d in dims], ["$%s$^2"%str(d) for d in dims])
    # plt.xticks([])

    plt.ylabel("square root of execution time (s)}")
    plt.savefig("timeVSdim.png")