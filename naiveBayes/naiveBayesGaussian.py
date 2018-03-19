#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-03-20"
__email__ = "cxweieee@126.com"

"""
Implementation naive bayes with continous variables
"""
import numpy as np

class NaiveBayesGaussian:
    """
    Implementation the naive bayes classifier with gaussian
    """
    def __init__(self):
        """
        """
        pass

    def train(self, X, Y):
        """
        @param X, np.array with shape (n, m) for n samples and each sample has m varibles
        @param Y, np.array with shape (n, ) for the class labels of each sample
        """
        self.classes = list(set(Y))
        n = len(Y)
        self.mean_X = {}
        self.var_X = {}
        self.prior = {}
        for one_class in self.classes:
            self.prior[one_class] = np.sum(Y==one_class) * 1.0 / n
            tmpX = X[Y==one_class]
            self.mean_X[one_class] = np.mean(tmpX, axis=0)
            self.var_X[one_class] = np.var(tmpX, ddof=1, axis=0)

    def predict(self, X):
        """
        """
        pred_Y = []
        for one_class in self.classes:
            score = np.exp(-np.power(X-self.mean_X[one_class], 2) / (2.*self.var_X[one_class])) / np.sqrt(2*np.pi*self.var_X[one_class])
            tmpY = list(np.prod(score, axis=1))
            pred_Y.append(tmpY)
        predIndex = np.argmax(pred_Y, axis=0)
        self.pred = [self.classes[index] for index in predIndex]
        return self.pred


        
            



