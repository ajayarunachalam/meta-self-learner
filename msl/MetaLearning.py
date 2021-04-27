
# coding: utf-8
__author__ = 'Ajay Arunachalam'
__version__ = '0.0.3'
__date__ = '27.4.2021'
# In[10]:


# load libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

class MetaEnsemble:
    NUM_CLASS = globals()

    def set_config(**kwargs):
        """
        Select NUMBER OF CLASS INPUT TO THE META ENSEMBLER MODEL
        """
        for key, value in kwargs.items():
            print("{0} = {1}".format(key, value))

        NUM_CLASS = list(kwargs.values())[0]

        return NUM_CLASS
       
    assert NUM_CLASS == NUM_CLASS

    # objective function for FIRST ENSEMBLE METHOD

    def objective_func_ensemble1(w, Xs, y, NUM_CLASS=NUM_CLASS): #n_class = int
        """
        Function to be minimized in the Ensemble1 ensembler.
        Parameters:
        ----------
        w: array-like, shape=(n_preds)
        Candidate solution to the optimization problem (vector of weights).
        Xs: list of predictions to combine
        Each prediction is the solution of an individual classifier and has a shape=(n_samples, n_classes).
        y: array-like shape=(n_samples,)
        Class labels
        n_class: int
        Number of classes in the problem (26 in LETTER RECOGNITION DATA)
        Return:
        ------
        score: Score of the candidate solution.
        """
        w = np.abs(w)
        solution = np.zeros(Xs[0].shape)
        for i in range(len(w)):
            solution += Xs[i] * w[i]
        # using log loss as objective function (any objective function can be used)
        score = log_loss(y, solution)
        return score

    # ensemble-1 ALGORITHM
    class Ensemble_one(BaseEstimator):
        """
        Given a set of predictions $X_1, X_2, ..., X_n$, it computes the optimal set of weights
        $w_1, w_2, ..., w_n$; such that minimizes $log\_loss(y_T, y_E)$,
        where $y_E = X_1*w_1 + X_2*w_2 +...+ X_n*w_n$ and $y_T$ is the true solution.
        """
        
        def __init__(self, NUM_CLASS): # n_class=NUM_CLASS
            super(MetaEnsemble.Ensemble_one, self).__init__()
            self.NUM_CLASS = NUM_CLASS
            
        def fit(self, X, y):
            """
            Learn the optimal weights by solving an optimization problem.
            Parameters:
            ----------
            Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has shape=(n_samples, n_classes).
            y: array-like
            Class labels
            """
            Xs = np.hsplit(X, X.shape[1] / self.NUM_CLASS) #self.n_class
            # Initial solution has equal weight for all individual predictions
            x0 = np.ones(len(Xs)) / float(len(Xs))
            # Weights must be bounded in [0,1]
            bounds = [(0,1)]* len(x0)
            # All weights must sum to 1
            cons = ({'type':'eq', 'fun': lambda w: 1-sum(w)})
            # calling the solver
            res = minimize(MetaEnsemble.objective_func_ensemble1,x0,args=(Xs,y, self.NUM_CLASS),  #self.n_class
                method='SLSQP',
                bounds=bounds,
                constraints=cons)
            self.w = res.x
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            
            """
            Use the weights learned in training to predict class probabilities.
            Parameters:
            ----------
            Xs: list of predictions to be blended.
            Each prediction is the solution of an individual classifier and has shape=(n_samples, n_classes).
            Return:
            ------
            y_pred: array_like, shape=(n_samples, n_class)
            The blended prediction.
            """
            Xs = np.hsplit(X, X.shape[1] / self.NUM_CLASS)
            y_pred = np.zeros(Xs[0].shape)
            for i in range(len(self.w)):
                y_pred += Xs[i] * self.w[i]
            return y_pred


# In[12]:


    # objective function for SECOND ENSEMBLE METHOD
    def objective_func_ensemble2(w, Xs, y, n_class=NUM_CLASS):


        """
        Function to be minimized in the Ensemble2 ensembler.
        Parameters:
        ----------
        w: array-like, shape=(n_preds)
        Candidate solution to the optimization problem (vector of weights).
        Xs: list of predictions to combine
        Each prediction is the solution of an individual classifier and has a shape=(n_samples, n_classes).
        y: array-like shape=(n_samples,)
        Class labels
        n_class: int
        Number of classes in the problem, i.e. = 26 in LETTER RECOGNITION DATA
        Return:
        ------
        score: Score of the candidate solution.
         """
        w_range = np.arange(len(w))%n_class
        for i in range(n_class):
            w[w_range==i] = w[w_range==i] / np.sum(w[w_range==i])
        solution = np.zeros(Xs[0].shape)
        for i in range(len(w)):
            solution[:,i % n_class] += Xs[int(i / n_class)][:,i % n_class] * w[i]
        # using log loss as objective function (any objective function can be used)
        score = log_loss(y, solution) 
        return score

    # Ensemble 2
    class Ensemble_two(BaseEstimator):
        """
        Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
        $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal
        set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes
        $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ...
        + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
        """
        def __init__(self, NUM_CLASS):
            super(MetaEnsemble.Ensemble_two, self).__init__()
            self.NUM_CLASS = NUM_CLASS

        def fit(self, X, y):
            """
            Learn the optimal weights by solving an optimization problem.
            Parameters:
            ----------
            Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has shape=(n_samples, n_classes).
            y: array-like
            Class labels
            """
            Xs = np.hsplit(X, X.shape[1] / self.NUM_CLASS) # self.n_class
            # Initial solution has equal weight for all individual preds
            x0 = np.ones(self.NUM_CLASS * len(Xs)) / float(len(Xs))
            # Weights must be bounded in [0,1]
            bounds = [(0,1)]*len(x0)
            #Calling the solver (constraints are directly defined in the objective
            #function)
            res = minimize(MetaEnsemble.objective_func_ensemble2, x0, args=(Xs, y, self.NUM_CLASS), # self.n_class
            method='L-BFGS-B',
            bounds=bounds,
            )
            self.w = res.x
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, X):
            """
            Use the weights learned in training to predict class probabilities.
            Parameters:
            ----------
            Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has shape=(n_samples, n_classes).
            Return:
            ------
            y_pred: array_like, shape=(n_samples, n_class)
            The ensembled prediction.
            """
            Xs = np.hsplit(X, X.shape[1]/self.NUM_CLASS)
            y_pred = np.zeros(Xs[0].shape)
            for i in range(len(self.w)):
                y_pred[:, i % self.NUM_CLASS] += Xs[int(i / self.NUM_CLASS)][:, i % self.NUM_CLASS] * self.w[i]
            return y_pred

