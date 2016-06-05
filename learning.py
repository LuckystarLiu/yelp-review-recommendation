# learn different models

import csv
import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve
from sklearn.preprocessing import PolynomialFeatures

def main():

    # load training and testing data set
    print('parsing training set...')
    X_train, y_train = parse('./data_set/train_set.csv')
    print('parsing testing set...')
    X_test, y_test = parse('./data_set/test_set.csv')
    print('train set: ', X_train.shape)
    print('test set: ', X_test.shape)

    # convert to polynomial features
    print('converting to polynomial features...')
    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train)
    X_test = poly.fit_transform(X_test)
    print('train set: ', X_train.shape)
    print('test set: ', X_test.shape)

    # learn with Linear Regression
    print('learning linear regression...')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    print('predicting linear regression...')
    y_pre = model.predict(X_test)
    print('Linear_regression: ')
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # SVR
    print('learning SVR...')
    model= svm.SVR(kernel='sigmoid', C=1)
    model.fit(X_train, y_train)
    print('predicting SVR...')
    y_pre = model.predict(X_test)
    print('SVM: ')
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Ridge
    print('learning Ridge...')
    model= linear_model.Ridge()
    model.fit(X_train, y_train)
    print('predicting Ridge...')
    y_pre = model.predict(X_test)
    print('Ridge: ')
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # ?
    print('learning ?...')
    model = linear_model.PassiveAggressiveRegressor()
    model.fit(X_train, y_train)
    print('predicting ?...')
    y_pre = model.predict(X_test)
    print('Poly: ')
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

def parse(filename):
    data_set = np.genfromtxt(filename, delimiter=',')
    X = data_set[:, 1:]
    y = data_set[:, 0]
    return X, y

main()
