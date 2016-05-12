# -*- coding: utf-8 -*-
"""
Created on Thu May 12 02:01:50 2016

@author: ZMP
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error,r2_score,roc_curve
import json
import re,csv
import cPickle
import numpy as np
import user_modified_json as uj
from sklearn import datasets, linear_model,svm
def parse(dir):
    data_set=[]
    with open(dir, 'rb') as csvfile:
        csvdata = csv.reader(csvfile, delimiter=',', quotechar=' ')
        for row in csvdata:
            temp=[]
            for i in range(len(row)):
                if(row[i]=='True'):
                    temp+=[1]
                elif(row[i]=='False'):
                    temp+=[0]
                else:
                    temp+=[float(row[i])]
            data_set+=[temp]
    y_train=[]
    x_train=[]
    for row in data_set:
        x_temp=[]
        for i in range(len(row)):
            if i==0:
                y_train+=[row[0]]
            else:
                x_temp+=[row[i]]
        x_train+=[x_temp]
    return x_train,y_train

def linear_regression():
    x_train,y_train=parse('data_set.csv')
    x_test,y_test=parse('data_set_test.csv')
    model= linear_model.LinearRegression()
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    print 'Linear_regression: '
    print 'mean absolute error: ', mean_absolute_error(y_test, y_pre)
    print 'r2_score: ', r2_score(y_test, y_pre)
def SVR():
    x_train,y_train=parse('data_set.csv')
    x_test,y_test=parse('data_set_test.csv')
    model= svm.SVR(kernel='sigmoid', C=1)
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    print 'SVM: '
    print 'mean absolute error: ', mean_absolute_error(y_test, y_pre)
    print 'r2_score: ', r2_score(y_test, y_pre)
def ridge():
    x_train,y_train=parse('data_set.csv')
    x_test,y_test=parse('data_set_test.csv')
    model= linear_model.Ridge()
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    print 'Ridge: '
    print 'mean absolute error: ', mean_absolute_error(y_test, y_pre)
    print 'mean', np.mean(y_test)
    print 'r2_score: ', r2_score(y_test, y_pre)
def poly():
    x_train,y_train=parse('data_set.csv')
    x_test,y_test=parse('data_set_test.csv')
    model= linear_model.PassiveAggressiveRegressor()
    model.fit(x_train,y_train)
    y_pre=model.predict(x_test)
    print 'Poly: '
    print 'mean absolute error: ', mean_absolute_error(y_test, y_pre)
    print 'r2_score: ', r2_score(y_test, y_pre)
linear_regression()
SVR()
ridge()
poly()