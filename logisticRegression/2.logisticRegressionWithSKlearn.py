#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

from readData import *
from sklearn.linear_model import SGDClassifier

X_train, y_train = readIris("data/iris_train.csv")
X_test, y_test = readIris("data/iris_test.csv")

lr = SGDClassifier(loss="log", penalty="none")
lr.fit(X_train, y_train)
acc = lr.score(X_test, y_test)

print "Predict Accuracy:", acc

