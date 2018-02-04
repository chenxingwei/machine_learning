#!/usr/bin/env python2.7
#-*- coding:utf-8 -*-

import os, glob

def readIris(infile):
    data = map(lambda x:x.strip().split(","),open(infile).readlines())
    X = []
    y = []
    names = "Iris-setosa"
    for line in data:
        tmp = [float(x) for x in line[:-1]]
        X.append(tmp)
        if line[-1] == names:
            y.append(1)
        else:
            y.append(0)
    return X, y



