# use only bag of words to learn models and compare their performance

import random
import json
import numpy as np
import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression, PassiveAggressiveRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.externals import joblib

def main():

    # read review data
    print('parsing review data...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')

    # use only reviews posted after 2008
    valid_reviews = []
    for review in reviews:
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        if review_date.year < 2008: 
            continue
        valid_reviews.append(review)
    reviews = valid_reviews

    # sample the data
    # sample_num = len(reviews)
    # print('sampling...', sample_num, 'out of', len(reviews))
    # reviews = sample(reviews, sample_num)

    # tokenize text for all reviews
    print('tokenizing text for all reviews...')
    texts = [review['text'] for review in reviews]
    count_vect = CountVectorizer(max_features = 100)
    X = count_vect.fit_transform(texts)

    # transform from occurrence to frequency
    print('converting occurrence to frequency...')
    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(X)

    # load the linear model for normalization
    clf = joblib.load('./normalization/linear_model_for_normalization.pkl')

    # get labels
    print('calculating labels...')
    y = []
    for review in reviews:
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        # normalize
        normalizor = clf.predict(np.array([[review_date.year]]))[0][0]
        review_quality = sum(review['votes'].values()) / normalizor
        y.append(review_quality)

    # splitting into train and test set
    print('splitting into train and test set...')
    train_len = int(X.shape[0] * 0.6)
    X_train = X[:train_len, :]
    y_train = y[:train_len]
    X_test = X[train_len:, :]
    y_test = y[train_len:]
    print('train size:', X_train.shape)
    print('test size:', X_test.shape)

    # convert to polynomial features
    # print('converting to polynomial features...')
    # poly = PolynomialFeatures(2)
    # X_train = poly.fit_transform(X_train.toarray())
    # X_test = poly.fit_transform(X_test.toarray())
    # print('train set: ', X_train.shape)
    # print('test set: ', X_test.shape)

    # scale the attributes to [0, 1]
    print('standardizing the features...')
    min_max_scaler = MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    # training classifiers
    print('training, predicting and evaluating...')

    # Dummy Regression (baseline model)
    print('\nDummy Regression:')
    model = DummyRegressor(strategy='mean')
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Linear Regression
    print('\nLinear_regression: ')
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Ridge
    print('\nRidge: ')
    model = Ridge()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # passive aggresive
    print('\nPoly: ')
    model = PassiveAggressiveRegressor()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # AdaBoost
    print('\nAdaBoost: ')
    model = AdaBoostRegressor()
    model.fit(X_train, y_train)
    y_pre = model.predict(X_test)
    print('mean absolute error: ', mean_absolute_error(y_test, y_pre))
    print('r2_score: ', r2_score(y_test, y_pre))

    # Random Forest
    print('\nRandom Forest:')
    model = RandomForestRegressor()
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
