#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC


class NBSVM:

    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        # Keep additional keyword arguments to pass to the classifier
        self.kwargs = kwargs

    def fit(self, X, y):
        f_1 = csr_matrix(y).transpose()
        f_0 = csr_matrix(np.subtract(1,y)).transpose() #Invert labels
        # Compute the probability vectors P and Q
        p_ = np.add(self.alpha, X.multiply(f_1).sum(axis=0))
        q_ = np.add(self.alpha, X.multiply(f_0).sum(axis=0))
        # Normalize the vectors
        p_normed = np.divide(p_, float(np.sum(p_)))
        q_normed = np.divide(q_, float(np.sum(q_)))
        # Compute the log-ratio vector R and keep for future uses
        self.r_ = np.log(np.divide(p_normed, q_normed))
        # Compute bayesian features for the train set
        f_bar = X.multiply(self.r_)
        # Fit the regressor
        self.lr_ = LogisticRegression(dual=True, **self.kwargs)
        self.lr_.fit(f_bar, y)

    def predict(self, X):
        return self.lr_.predict(X.multiply(self.r_))

