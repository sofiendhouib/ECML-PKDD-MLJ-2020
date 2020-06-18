#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join as pathJoin
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from tempfile import mkdtemp
from sys import path as pathSys
import myIOFuncs as ioFuncs
import pickle


# =============================================================================
# Scikit-learn imports
# =============================================================================
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA


# =============================================================================
# Metric/Similarity learning modules: Ours, SLLC, ITML, LMNN
# =============================================================================
import similarity_learning as sl
from metric_learn import LMNN, ITML_Supervised

plt.close('all')
cachedir = mkdtemp()


# %%=============================================================================
# Loading data
# =============================================================================
"""
    Tested data sets: blobs, breast_cancer, ionosphere, pima, splice, svmguid1, cod-rna
    splice, svmguide1 and cod-rna have a predefined test set
"""
dataPath = "data/" # where data is stored

datasetDict ={  
                'blobs': 'blobs.data', # a toy set, to check if the script works well
                # 'splice': 'splice.train',
                # 'svmguide1': 'svmguide1.train',
                # 'cod-rna': 'cod-rna.train',
                # 'breast': 'breast.data',
                # 'ionosphere': 'ionosphere.data',
                # 'pima': 'pima.data',
            }

#%%
"""
    Defining different pipelines for each metric learning algorithm:
        * metric learning algorithm: transforms the data after learning the metric
        * classifier: l1 linear classifier
"""

l1LinClf = sl.l1LinearClassifier(solver= 'cvxpy')
lambdaRange = np.logspace(2,-3,6)
l1LinClfParamGrid = {"linear__lambda_reg": lambdaRange}

# 
metricLearnersDict= {
                    'closed-form': sl.bilinearSimilarityLearner(algorithm= 'closed-form'),
                    # 'RVML': sl.RVMLSimilarityLearner(kernel= 'rbf', VP= 'classBased'),
                    # 'SLLC': sl.bilinearSimilarityLearner(algorithm= 'sllc'),
                    # 'LMNN': LMNN(),
                    # 'ITML': ITML_Supervised(), 
                }
def pipelineConstructor(algoName):
    classifierStep = ('linear', l1LinClf)
    return Pipeline(steps = [(algoName, metricLearnersDict[algoName]), classifierStep])
     
    
estimatorsDict= {}
for algoName in metricLearnersDict.keys():
    estimatorsDict[algoName] = {"estimator": pipelineConstructor(algoName)}

estimatorsDict["closed-form"]["param_grid"] = {**{"closed-form__beta_reg": np.logspace(4,-7,12)}, **l1LinClfParamGrid}
# estimatorsDict["SLLC"]["param_grid"] = {**{"SLLC__beta_reg": np.logspace(-2,-7,6), "SLLC__gamma": np.logspace(-2,-7,6)}, **l1LinClfParamGrid}
# estimatorsDict["RVML"]["param_grid"] = {**{"RVML__l": [10**p for p in range(-5,2)]}, **l1LinClfParamGrid} #exactly like in the authors' code
# estimatorsDict["LMNN"]["param_grid"] = {}
# estimatorsDict["ITML"]["param_grid"] = {**{"ITML__gamma": np.logspace(-4,4,9)}}


#%%
""" 
    Learning different classifiers after cross validation: 
    loop over data sets and different algorithms, with 100 runs for each combination
"""

nbSplits = 10 # 
for (dataName, dataFile) in list(datasetDict.items()):
    # =============================================================================
    #  Loading the data set: train and test sets   
    # =============================================================================
    print("\n")
    
#    Case where a predefined train/test split exists
    if dataFile.endswith('train'):
        X_train, y_train = ioFuncs.loadFromTxt(pathJoin(dataPath, dataFile))
        if len(X_train)>=10000: # if the dataset is too large, take a 10% fraction, that has a representative class repartition (for memory issues)
            _, X_train, _, y_train = model_selection.train_test_split(X_train, y_train, test_size=0.1, stratify= y_train)
        X_test, y_test = ioFuncs.loadFromTxt(pathJoin(dataPath, dataFile[:-5]+'test'), idInd= None, classInd= 0)
        
        X, y = np.vstack((X_train, X_test)), np.hstack((y_train, y_test))
        testCV = model_selection.PredefinedSplit([-1]*len(X_train) + [0]*len(X_test)) # predefined split specification for sklearn's functions
        
#     case with a manual random train/test split, done 100 times
    else:
        X, y = ioFuncs.loadFromTxt(pathJoin(dataPath, dataFile))
        testCV = model_selection.ShuffleSplit(n_splits= nbSplits, test_size= 0.3)
    try:    
        sigmaSquared = np.mean(pdist(X, metric= "sqeuclidean"))
    except MemoryError:
        sigmaSquared = np.mean(pdist(X_train, metric= "sqeuclidean"))
        
        
    dataRange = (-1/np.sqrt(X.shape[1]), 1/np.sqrt(X.shape[1])) #[-1/sqrt(d), 1/sqrt(d)]
    minMaxScaler = preprocessing.MinMaxScaler(feature_range= dataRange)
    
#    estimatorsDict["ITML"]["estimator"].steps[0] = ITML_Supervised(num_constraints= int(0.7*len(X))) # special case of ITML: number of constraints = number of landmarks
    for algoName in list(estimatorsDict.keys()):
        if algoName == 'RVML':
            dataTransformingPipeline = Pipeline([('minMaxScaler', minMaxScaler)])
        else: # if not RVML, use KPCA
            nKPCA = 4*X.shape[1] if X.shape[1]<8 else 3*X.shape[1]
            kernelPCA = KernelPCA(kernel= 'rbf', n_components= nKPCA, gamma= 0.5/sigmaSquared)
            dataTransformingPipeline = Pipeline([
                                        ('kernel_pca',  kernelPCA),
                                        ('minMaxScaler', minMaxScaler),
                                        ])
    
        print("data set: %s, algorithm: %s" %(dataName, algoName))
        algorithmDict = estimatorsDict[algoName]
        
        # add number of constraints to ITML
        if algoName == "ITML":
            try:
                algorithmDict["param_grid"]["ITML__num_constraints"] = len(X_train)
            except NameError:
                algorithmDict["param_grid"]["ITML__num_constraints"] = int(0.7*len(X))
        # =============================================================================
        #   Performe a grid search on train/validation, then depending whether there are predefined train/test, use them or average over 100 runs
        # =============================================================================
    
        gridSearcher = model_selection.GridSearchCV(estimator= Pipeline(dataTransformingPipeline.steps + algorithmDict["estimator"].steps), 
                                                        param_grid= algorithmDict["param_grid"],
                                                        n_jobs= -1, verbose= 0, cv= model_selection.KFold(n_splits= 5))
        
        if algoName == "ITML":
            cvResult = model_selection.cross_validate(gridSearcher, X, (y==1).astype(np.int16), n_jobs= 1, cv= testCV, verbose= 2)
        else:
            cvResult = model_selection.cross_validate(gridSearcher, X, y, n_jobs= 1, cv= testCV, verbose= 2)
        
        # Choose a name to save cross validaiton result
        with open('%s-%s-KPCA.pickle'%(dataName, algoName), 'wb') as handle:
            pickle.dump(cvResult, handle, protocol=pickle.HIGHEST_PROTOCOL)
