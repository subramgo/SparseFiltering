# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:35:22 2014

@author: gsubramanian
"""
import numpy as np
from scipy.optimize import minimize

epsilon = 1e-8


def soft_absolute(v):
    return np.sqrt(v**2 + epsilon)


def get_objective_fn(X,n_dim,n_features):
    def _objective_fn(W):
        W = W.reshape(n_dim,n_features)
        Y = np.dot(X,W)
        Y = soft_absolute(Y)
        
        # Normalize feature across all examples
        # Divide each feature by its l2-norm
        Y = Y / np.sqrt(np.sum(Y**2,axis=0) + epsilon)        
        
        # Normalize feature per example
        Y = Y / np.sqrt(np.sum(Y**2,axis=1)[:,np.newaxis] + epsilon )
        
        return np.sum(Y)
    return _objective_fn


def sfiltering(X,n_features=5):
    n_samples,n_dim = X.shape
    # Intialize the weight matrix W (n_dim,n_features)
    # Intialize the bias term b(n_features)
    W = np.random.randn(n_dim,n_features)
    obj_function = get_objective_fn(X,n_dim,n_features)
    
    opt_out = minimize(obj_function,W,method='L-BFGS-B',options={'maxiter':10,'disp':True})
    W_final = opt_out['x'].reshape(n_dim,n_features)
    
    transformed_x = np.dot(X,W_final)
    return transformed_x
    









