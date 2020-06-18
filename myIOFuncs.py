#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt

""" For reading a data set from the UCI machine learning repository"""
def loadFromTxt(fname, idInd= None, classInd= 0):
    data = np.loadtxt(fname, delimiter = ',')
    featureInds = [i for i in range(data.shape[1]) if not(i in [classInd, idInd])]
    X = data[:,featureInds]
    y = data[:,classInd]
    classes = np.unique(y)
    if len(classes) != 2: raise ValueError("data set has more than two classes")
    return X, 2/(classes[1] - classes[0])*(y - classes[1])+1

#%%


""" Used with Scikit-Learn's learning_curve'
Plots learning curves with standard deviation over different runs specified
by the cross validation strategy given to learning_curve"""
def plot_learning_curve(train_sizes, train_scores, test_scores, ylim= None, title= "Learning curve"):
    plt.figure(figsize= (16,9))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.semilogy(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.semilogy(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plotWithConf(xArray, yArray, axis, interval= [0.05, 0.95]):
    meanYArray= np.mean(yArray, axis=  axis)
    # stdYArray= np.std(yArray, axis= axis)
    quantile1Array = np.quantile(yArray, q= interval[0], axis= axis)
    quantile2Array = np.quantile(yArray, q= interval[1], axis= axis)
    plt.plot(xArray, meanYArray)
    # plt.fill_between(xArray, meanYArray - stdYArray, meanYArray + stdYArray, alpha= 0.3)
    plt.fill_between(xArray, quantile1Array, quantile2Array, alpha= 0.3)
    return None

