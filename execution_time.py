#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    This script compares the time required to learn a metric by SLLC, LMNN, ITML,  RVML and closed-form (our algorithm)
    Its current setting compares RVML vs our algorithm on the cod-rna data set
"""
from os.path import join as pathJoin
import numpy as np
from matplotlib import pyplot as plt
from tempfile import mkdtemp
import pandas as pd
from time import time
from sys import path as pathSys
pathSys.append('../')
import myIOFuncs as ioFuncs

# =============================================================================
# Scikit-learn imports
# =============================================================================
from sklearn.model_selection import train_test_split

# =============================================================================
# Metric/Similarity learning modules: Ours, SLLC, ITML, LMNN
# =============================================================================
import similarity_learning as sl # for our algorithm and SLLC
from metric_learn import LMNN, ITML_Supervised

plt.close('all')
cachedir = mkdtemp()


# %%=============================================================================
# Loading data
# =============================================================================
"""
    Tested data sets: breast_cancer_wisconsin, ionosphere, pima, splice, svmguid1, cod-rna
    splice, svmguide1 and cod-rna have a predefined test set
"""
dataPath = "data/" # where data is stored

datasetDict ={  
                'blobs': 'blobs.data', # a toy set, to check if the script works well
                'breast': 'breast.data',
#                'ionosphere': 'ionosphere.data',
#                'pima': 'pima.data',
#                'splice': 'splice.train',
#                'svmguide1': 'svmguide1.train',
                # 'cod-rna': 'cod-rna.train',
            }

#%% similarity/metric learners
metricLearnersDict= {                   
                    'SLLC' : sl.bilinearSimilarityLearner(algorithm= 'SLLC'),
                    'LMNN' : LMNN(use_pca= False),
                    'ITML' : ITML_Supervised(num_constraints= 1),
                    'RVML': sl.bilinearSimilarityLearner(algorithm= 'RVML'),
                    'closed-form': sl.bilinearSimilarityLearner(algorithm= 'closed-form'),
                }

#%%
""" 
    Computing execution times: 
    loop over data sets and different algorithms, with 100 runs for each combination
"""
nbRepetitions = 10
execTime = np.zeros((len(datasetDict.keys()), len(metricLearnersDict.keys()), nbRepetitions))
for i, (dataName, dataFile) in enumerate(datasetDict.items()):
    print("\n")
    X, y = ioFuncs.loadFromTxt(pathJoin(dataPath, dataFile), idInd= None, classInd= 0)
    
    if len(X)>=10000: # if the dataset is too large, take a 10% fraction, that has a representative class repartition (for memory issues)
        _, X, _, y = train_test_split(X, y, test_size=0.1, stratify= y)
    
    for j, (metricLearnerName, metricLearner) in enumerate(metricLearnersDict.items()):
        metricLearnerArgs= (X, (y+1)/2) if metricLearnerName == 'ITML' else (X,y)
        print("data set: %s, algorithm: %s" %(dataName, metricLearnerName))

        for k in range(nbRepetitions):
            print(k)
            startTime= time()
            metricLearner.fit(*metricLearnerArgs)
            endTime= time()
            execTime[i, j, k]= endTime - startTime

#%% Mean execution times as a table
np.save("execTime", execTime)
execTimeDf = pd.DataFrame(data= execTime.mean(axis= -1).T, index= metricLearnersDict.keys(), columns= datasetDict.keys())
execTimeDf.to_latex("execTime.tex")

#%% Plotting execution times
execTime = np.load("execTime.npy")
execTimeFlat = np.log10(execTime.flatten())
execTimeFlatDf= pd.DataFrame(data = execTimeFlat - np.min(execTimeFlat), columns= ["computation time"])
execTimeFlatDf["tuple_inds"] = [np.unravel_index(i, dims= (execTime.shape)) for i in execTimeFlatDf.index]
execTimeFlatDf["data set"] = execTimeFlatDf["tuple_inds"].apply(lambda x: list(datasetDict.keys())[x[0]])
execTimeFlatDf["algorithm"] = execTimeFlatDf["tuple_inds"].apply(lambda x: list(metricLearnersDict.keys())[x[1]])
execTimeFlatDf.to_csv("execTimeFlatDf.csv")

#%%
from seaborn import barplot
barplot(x= "data set", y= "computation time", hue= "algorithm", data= execTimeFlatDf, capsize= 0.1)
