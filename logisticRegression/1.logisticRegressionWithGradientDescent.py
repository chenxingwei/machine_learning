#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

from readData import *
from logistic_regression import logistic_regression_GD

X_train, y_train = readIris("data/iris_train.csv")
X_test, y_test = readIris("data/iris_test.csv")

lr = logistic_regression_GD()
lr.train(X_train, y_train)
lr.predict(X_test, y_test)

print "Predict Accuracy:", lr.accuracy

