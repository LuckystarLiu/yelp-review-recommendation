# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 00:32:08 2016

@author: ZMP
"""

# learn different models

import csv
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.dummy import DummyRegressor,DummyClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import cross_validation
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
#from sklearn.neural_network import MLPClassifier

def main():

    # load training and testing data set
    print('parsing data set...')
    X, y = parse('./data_set/classifications_dataset.csv')
    print('train set: ', X.shape)
    X=X[:200000,:]
    y=y[:200000,:]
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
    

    # The result turns out to be worse using non-linear polynomial regression
    # convert to polynomial features
    # print('converting to polynomial features...')
    # poly = PolynomialFeatures(2)
    # X_train = poly.fit_transform(X_train)
    # X_test = poly.fit_transform(X_test)
    # print('train set: ', X_train.shape)
    # print('test set: ', X_test.shape)

    # scale the attributes to [0, 1]
    print('standardizing the features...')
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # training classifiers
    print('training, predicting and evaluating...')

    # Dummy Classifier (baseline model)
    print('\nDummy Classification: (baseline)')
    model = DummyClassifier(strategy='uniform')
    model.fit(X_train, y_train)
    #print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('acc score: ', model.score(X_test, y_test))

    # DecisionTreeClassifier
    print('\nDecisionTreeClassifier: ')
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    print('acc score: ', model.score(X_test, y_test))

    # KNN Regression
    # print('\nKNN Regression: ')
    # model = KNeighborsRegressor()
    # model.fit(X_train, y_train)
    # y_pre = model.predict(X_test)
    # print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    # print('r2_score: ', r2_score(y_test, y_pre))

    # Neural Network - Bernoulli Restricted Boltzmann Machine (RBM)
    # print('\nNeural Network - RBM: ')
    # model = BernoulliRBM()
    # model.fit(X_train, y_train)
    # y_pre = model.predict(X_test)
    # print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    # print('r2_score: ', r2_score(y_test, y_pre))



    # KNeighborsClassifier
    print('\nKNeighborsClassifier: ')
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    #print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('acc score: ', model.score(X_test, y_test))
    
    #GaussianNB()
    print('\nGaussianNB(): ')
    model = GaussianNB()
    model.fit(X_train, y_train)
    #print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('acc score: ', model.score(X_test, y_test))
    
    
    # MLPClassifier
    #print('\nMLPClassifier: ')
    #model = MLPClassifier()
    #model.fit(X_train, y_train)
    #print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    #print('acc score: ', model.score(X_test, y_test))

def parse(filename):
    data_set = np.genfromtxt(filename, delimiter=',')
    X = data_set[:, 1:]
    y = data_set[:, 0]
    return X, y

main()
