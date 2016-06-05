# use only bag of words to learn models and compare their performance

import random
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, linear_model, svm
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve

def main():

    # read review data
    print('parsing review data...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')

    # sample the data
    print('sampling...')
    reviews = sample(reviews, 10000)

    # tokenize text for all reviews
    print('tokenizing text for all reviews...')
    texts = [review['text'] for review in reviews]
    count_vect = CountVectorizer(max_features = None)
    X = count_vect.fit_transform(texts)

    # transform from occurrence to frequency
    print('converting occurrence to frequency...')
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)

    # get labels
    print('calculating labels...')
    y = [sum(review['votes'].values()) for review in reviews]

    # splitting into train and test set
    print('splitting into train and test set...')
    train_len = int(len(reviews) * 0.6)
    X_train = X[:train_len, :]
    y_train = y[:train_len]
    X_test = X[train_len:, :]
    y_test = y[train_len:]

    # training a classifier
    print('training, predicting and evaluating...')

    # naive bayes
    print('Naive Bayes: ')
    clf = MultinomialNB().fit(X_train, y_train)
    y_pre = clf.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Linear Regression
    print('Linear_regression: ')
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # SVM
    print('SVM: ')
    model= svm.SVR(kernel='sigmoid', C=1)
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Ridge
    print('Ridge: ')
    model= linear_model.Ridge()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # passive aggresive
    print('Poly: ')
    model = linear_model.PassiveAggressiveRegressor()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

def parse_json(filename):
    data = []
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data

def sample(data, num):
    num = min(num, len(data))
    data_indices = list(range(len(data)))
    random.shuffle(data_indices)
    return [data[i] for i in data_indices[:num]]

main()
