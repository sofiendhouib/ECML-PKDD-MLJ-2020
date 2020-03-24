#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script plots learning curves of the beta parameter of our algorithm
"""

import numpy as np
from matplotlib import pyplot as plt
from similarity_learning import bilinearSimilarityLearner, l1LinearClassifier
from sklearn.model_selection import validation_curve, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer, make_blobs

X, y = load_breast_cancer(return_X_y = True)
#X, y = make_blobs(n_samples= 1000, centers= 2)


betaRange = 10.0**np.arange(-6, 6)
lambda_reg = 1

estimator = Pipeline(steps = [('similarity', bilinearSimilarityLearner()), 
                              ('classifier', l1LinearClassifier(solver='cvxpy', lambda_reg= lambda_reg))]) # by default, the used algorithm is ours

train_scores, test_scores = validation_curve(estimator, X, y, param_name="similarity__beta_reg",
                                             param_range= betaRange, cv= RepeatedKFold(n_repeats= 1), n_jobs= -1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve")
plt.xlabel("beta")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(betaRange, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(betaRange, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(betaRange, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(betaRange, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.show()
