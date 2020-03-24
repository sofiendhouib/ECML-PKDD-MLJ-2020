#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.base import BaseEstimator, TransformerMixin
import cvxpy as cvx
import numpy as np
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sys import path as pathSys


"""This import is to include the RVML algorithm (https://perso.univ-st-etienne.fr/pem82055/RegressiveVirtualMetricLearning.html)"""
pathSys.append('./RVML/') # path where RVML folder is stored (Available for download on the above URL)
import VirtualPoints as vp
import RVML as rvml


class bilinearSimilarityLearner(BaseEstimator, TransformerMixin):  
    """An example of classifier"""

    def __init__(self, gamma= 1, beta_reg= 1e-3, is_similarity= True, algorithm= 'closed-form'):
        """
        Called when initializing the classifier
        """
        self.paramMat_ = np.array([], dtype= np.float64)
        self.landmarks_ = np.array([], dtype= np.float64)
        self.landmarks_labels_ = np.array([], dtype= np.float64)
        self.set_params(**{'gamma': gamma, 'beta_reg': beta_reg, 'is_similarity': is_similarity, 'algorithm': algorithm})
        self.epsilon_ = -np.inf
        self.is_similarity = is_similarity


    def fit(self, X, y):
        params = self.get_params()
        self.paramMat_ = self.__learnSimMatrix(X, y, params['beta_reg'], params['gamma'])
        self.landmarks_ = X.copy()
        self.landmarks_labels_ = y.copy()
        return self
    
    def transform(self, X, landmarks= None):
        if landmarks == None:
            landmarks = self.landmarks_.copy()
        simMat = computeBilinSimMat(X, landmarks, self.paramMat_)
        return simMat if self.is_similarity else -simMat
    
    """margin violation score = 1 - margin violation error"""    
    def score(self, X, y, landmarks= None):
        if landmarks == None:
            landmarks = self.landmarks_
        simMat = computeBilinSimMat(X, landmarks, self.paramMat_)
        gamma = self.get_params()['gamma']
        margins = np.mean(y[:,None]*self.landmarks_labels_*simMat, axis= 1)
        return np.mean(margins >= gamma)
        
#        
    def __learnSimMatrix(self, X, y, beta_reg, gamma):
        n, d = X.shape
        algorithm = self.get_params()['algorithm'].lower()

        """The following case corresponds to linear RVML. It is only used to allow fair computation time comparison between the two methods"""
        if algorithm == 'rvml':
            V = vp.classBasedVP(X, y)
            L = np.linalg.inv(X.T.dot(X) + beta_reg*n*np.eye(d)).dot(X.T).dot(V) # a copy-past of the expression in the original linear RVML function
            return L
        
        """ Implementation of SLLC"""
        
        if algorithm == 'sllc':
            X_signed = y[:, np.newaxis]*X
            A = cvx.Variable((d,d))
            loss = cvx.sum(cvx.pos(1-X_signed*A*cvx.sum(X_signed, axis= 0).T/(n*gamma)))/n
            reg = beta_reg*cvx.sum_squares(A)
                
            prob = cvx.Problem(objective= cvx.Minimize(loss + reg))
            prob.solve(solver= cvx.MOSEK)
            return np.array(A.value)
        
        """ our algorithm"""
        if algorithm == 'closed-form':
            mu = np.dot(X.T,y)/n
            sqNormMu = np.dot(mu, mu)
            
            Sigma = np.dot(X.T, X)/n
            
            betaEq = beta_reg/sqNormMu
    
            s  = np.linalg.eigvalsh(Sigma)[0] + betaEq # bound on s
            
            return s/sqNormMu*np.outer(np.linalg.solve(Sigma + betaEq*np.eye(d), mu), mu)
        
from scipy.spatial.distance import  pdist
from sklearn.metrics.pairwise import rbf_kernel


"""RVML class"""
class RVMLSimilarityLearner(BaseEstimator, TransformerMixin):  
    """
        This class implements the RVML algorithm.
        It is based on both the paper and the code available on the link below:
            https://perso.univ-st-etienne.fr/pem82055/RegressiveVirtualMetricLearning.html
    """

    def __init__(self, l= 1, VP = 'classBased', reg= 0.5, eta= 10, kernel= 'rbf'):
        """
        Called when initializing the classifier
        """
        self.paramL_ = np.array([], dtype= np.float64)
        self.paramV_ = np.array([], dtype= np.float64)
        self.paramX_ = np.array([], dtype= np.float64)
        self.set_params(**{'l':l, 'VP': VP, 'eta': eta, 'reg': reg, 'kernel': kernel})
        self.sigmaSq= None

    def fit(self, X, y):
        V = self.__computeVP(X,y)
        self.paramV_ = V.copy()
        if self.get_params()['kernel'] == 'linear':
            self.paramL = self.__learnL(X, V)
        else :
            self.sigmaSq = np.mean(pdist(X, metric= "sqeuclidean"))
            self.paramX_  = X.copy()
        return self
    
    def transform(self, X):
        params = self.get_params()
        if params['kernel'] == 'linear':
            XL = np.dot(X, self.paramL_)
            return XL
        else:
            X_l = self.paramX_.copy()
            n = X_l.shape[0]
            sigmaSq = self.sigmaSq
            gammaRbf = 0.5/sigmaSq
            KL = np.linalg.inv(rbf_kernel(X_l,X_l, gamma= gammaRbf)+params['l']*n*np.identity(n)).dot(self.paramV_)
            return rbf_kernel(X,X_l, gamma= gammaRbf).dot(KL)
            
    
    """ 
        computes the virtual points, depending on the method: class based or optimal transport based
    """
    def __computeVP(self, X,y):
        params= self.get_params()
        if params['VP']== 'transportBased':
            return vp.transportBasedVP(X, y, reg= params['reg'], eta= params['eta'])
        elif params['VP']== 'classBased':
            return vp.classBasedVP(X,y)        
    
    """
        Computes the linear transformation L
    """
    def __learnL(self, X,V):
        n = X.shape[0]
        d = X.shape[1]
        l = self.get_params()['l']
        if d <= n:
            L = np.linalg.inv(X.T.dot(X) + l*n*np.eye(d)).dot(X.T).dot(V)
        else:
            L = X.T.dot(np.linalg.inv(X.dot(X.T) + l*n*np.identity(n))).dot(V)
        return L     
        

"""linear SVM with hinge loss and l1 penalization"""
class l1LinearClassifier(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
    
    def __init__(self, lambda_reg= 1e-3, solver= 'cvxpy'):

        self.set_params(**{'lambda_reg': lambda_reg, 'solver': solver})
        self.coef_ = np.array([])
        self.classes_ = np.array([])
        self.__logReg = LogisticRegression()
        self.fit_intercept = False
        self.intercept_ = 0


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.

        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """
        self.classes_= np.unique(y)
        params = self.get_params()
        self.coef_ = self.__learnSeparator(X, y, params['lambda_reg'], params['solver']).reshape(1, X.shape[1])
#        self.sparsify()
        return self
    
    def predict_proba(self, X):
        score = self.decision_function(X)
        self.__logReg.fit(score[:, np.newaxis], (score > 0).astype(np.int64))
        return self.__logReg.predict_proba(score[:, np.newaxis])[:,1]
    
    def __learnSeparator(self, X, y, lambda_reg, verbose= False):
        solverArg = self.get_params()['solver']
    
        if solverArg == "cvxpy":
#            print("cvxpy used")
            alphaVar = cvx.Variable(X.shape[1])
            loss = cvx.sum(cvx.pos(1 - y[:, np.newaxis]*X*alphaVar))
            reg = lambda_reg*cvx.norm1(alphaVar)
            prob = cvx.Problem(objective= cvx.Minimize(loss + reg))
            try:
                prob.solve(solver = cvx.MOSEK)
                alpha = np.array(alphaVar.value).flatten()
            except :
                print("Warning: MOSEK solver failed, switching to SGD...")
                solverArg = "sgd"
        
        # stochastic gradient descent: for large data sets        
        elif solverArg == "sgd":
#            print("sgd used")
            sgdLinClassifier= SGDClassifier(loss= 'hinge', penalty= 'l1', 
                                            alpha= lambda_reg, tol= 1e-30, verbose = False, 
                                            learning_rate= 'optimal', eta0= 1e-6, shuffle= True,
                                            max_iter= np.ceil(10**6 / len(y)), # empirical rule of thumb, given in the sklearn documentation
                                            average= 10,
                                            )
            
            sgdLinClassifier.fit(X, y)
            alpha = sgdLinClassifier.coef_.flatten()
        
        return alpha
    
#%% functions used above

def hingeLoss(x):
    return np.maximum(0, 1-x)

def meanLoss(simMat, w1,w2, gamma):
    return np.mean(hingeLoss(np.dot(w1[:,None]*simMat, w2)/(gamma*simMat.shape[1])))

def computeBilinSimMat(X1, X2, A):
    return np.linalg.multi_dot((X1, A, X2.T))

