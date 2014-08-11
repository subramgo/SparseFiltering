# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:14:02 2014

@author: gsubramanian
"""
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from SparseFilter import *

def load_data():
    X,Y = make_classification(n_samples = 500,n_features=100)
    return X,Y


def simple_model(X,Y):
    clf_org_x = SVC()
    clf_org_x.fit(X,Y)
    predict = clf_org_x.predict(X)
    acc=  accuracy_score(Y,predict)
    return acc
    
    
X,Y = load_data()
acc = simple_model(X,Y)

X_trans = sfiltering(X,25)

acc1= simple_model(X_trans,Y)

X_trans1 = sfiltering(X_trans,10)

acc2= simple_model(X_trans1,Y)

print "Without sparsefiltering, accuracy = %f "%(acc)
print "One Layer Accuracy, = %f, Increase = %f"%(acc1,acc1-acc)
print "Two Layer Accuracy,  = %f, Increase = %f"%(acc2,acc2-acc1)