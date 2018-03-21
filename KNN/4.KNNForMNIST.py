#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

__author__ = "CHEN Xingwei"
__date__ = "2018-03-22"
__email__ = "cxweieee@126.com"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train_labels = np.argmax(mnist.train.labels, axis=1)
validation_labels = np.argmax(mnist.validation.labels, axis=1)
test_labels = np.argmax(mnist.test.labels, axis=1)
print "KNN"

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(mnist.train.images[:1000], train_labels[:1000])

validation_pred = neigh.predict(mnist.validation.images)
test_pred = neigh.predict(mnist.test.images)

validation_accuracy = np.sum(validation_labels==validation_pred) *\
    1.0 / len(validation_labels)

test_accuracy = np.sum(test_labels==test_pred) *\
    1.0 / len(test_labels)

print "Validation Accuracy: ", validation_accuracy * 100, "%"
print "Test Accuracy: ", test_accuracy * 100, "%"
