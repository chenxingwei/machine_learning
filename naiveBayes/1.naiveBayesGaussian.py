#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-03-20"
__email__ = "cxweieee@126.com"

"""
Implementation naive bayes with continous variables
"""
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

# Sklearn
from sklearn.naive_bayes import GaussianNB
f = GaussianNB()
f.fit(iris.data, iris.target)
pred = f.predict(iris.data)
print pred


# Numpy
from naiveBayesGaussian import NaiveBayesGaussian
f = NaiveBayesGaussian()
f.train(iris.data, iris.target)
pred = f.predict(iris.data)
print pred


