#!/usr/bin/env python2.7

import numpy as np

class NearestNeighbor:
  def __init__(self):
    pass
  
  def train(self, X, y):
    """
    @param X, np.array, with shape (n, m) for n samples,
    each sample have m features
    @param y, np.array, with shape (n, ) for n samples.
    """
    self.train_X = X
    self.train_y = y
    
  def predict(self, X):
    """
    @param X, np.array, with shape (n, m) for n samples,
    each sample have m features
    """
    num_test = X.shape[0]
    Ypred = np.zeros(num_test, dtype = self.train_y.dtype)
    
    for i in xrange(num_test):
      distances = np.sum(np.abs(self.train_X - X[i, :]), axis = 1)
      min_index = np.argmin(distances)
      Ypred[i] = self.train_y[min_index]
    return Ypred
  
