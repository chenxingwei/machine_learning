import pylab
import numpy as np
import tensorflow as tf
import os, glob, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from readData import *

train_X, train_y = readData("data/iris_train.csv")
test_X, test_y = readData("data/iris_test.csv")

n = len(train_X)
learning_rate = 0.001 

epochs = 30

X = tf.placeholder(tf.float32,[None,4])
y = tf.placeholder(tf.float32, [None,1])

W = tf.Variable(tf.zeros([4, 1])) + 0.1
b = tf.Variable(tf.zeros([1])) + 0.1
pred = 1.0 / (1.0+tf.exp(-(tf.matmul(X,W)+b)))

cost = tf.reduce_sum(-y*tf.log(pred)-(1.0-y)*tf.log(1-pred))
#cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


ss,ys = [],[]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in xrange(epochs):
        sess.run(optimizer, feed_dict={X:train_X,y:train_y})
        #print(sess.run(y, feed_dict={X:train_X,y:train_y}))
        #print(sess.run(pred,feed_dict={X:train_X,y:train_y}))
        #print(sess.run(W))
        #print(sess.run(b))
        #print(sess.run(cost, feed_dict={X:train_X,y:train_y}))
        



