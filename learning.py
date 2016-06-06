# learn different models

import csv
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression, PassiveAggressiveRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():

    # load training and testing data set
    print('parsing training set...')
    X_train, y_train = parse('./data_set/train_set.csv')
    print('parsing testing set...')
    X_test, y_test = parse('./data_set/test_set.csv')
    print('train set: ', X_train.shape)
    print('test set: ', X_test.shape)

    # The result turns out to be worse using this
    # convert to polynomial features
    # print('converting to polynomial features...')
    # poly = PolynomialFeatures(2)
    # X_train = poly.fit_transform(X_train)
    # X_test = poly.fit_transform(X_test)
    # print('train set: ', X_train.shape)
    # print('test set: ', X_test.shape)

    # training classifiers
    print('training, predicting and evaluating...')

    # Dummy Regression (baseline model)
    print('Dummy Regression: (baseline)')
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Ridge
    print('Ridge: ')
    model = Ridge()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Linear Regression
    print('Linear_regression: ')
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Passive aggresive
    print('Poly: ')
    model = PassiveAggressiveRegressor()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

def parse(filename):
    data_set = np.genfromtxt(filename, delimiter=',')
    X = data_set[:, 1:]
    y = data_set[:, 0]
    return X, y

main()
