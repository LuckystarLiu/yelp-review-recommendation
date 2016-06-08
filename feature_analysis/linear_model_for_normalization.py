# train the linear model on the reviews' votes by its time after 2008

import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib # save model

def main():

    # load reviews data
    print('parsing...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')

    # extract duration of post date and votes count
    print('extracting...')
    vote_n_count = []
    start_year = 2008
    for review in reviews:
        # review = reviews[i]
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        if review_date.year < start_year:
            continue
        index = review_date.year - start_year
        while len(vote_n_count) - 1 < index:
            vote_n_count.append({'votes': 0, 'count': 0})
        vote_n_count[index]['votes'] += sum(review['votes'].values())
        vote_n_count[index]['count'] += 1

    # average the votes for each year
    avg_votes = []
    for item in vote_n_count:
        if item['count'] == 0:
            avg_votes.append(0)
        else:
            avg_votes.append(item['votes'] / item['count'])

    # train linear model on average votes and year
    years = list(range(start_year, start_year+len(avg_votes)))
    X = np.array(years).reshape((len(years), 1))
    y = np.array(avg_votes).reshape((len(avg_votes), 1))
    print(X.shape, y.shape)
    clf = LinearRegression()
    clf.fit(X, y)

    # save the linear model
    joblib.dump(clf, 'linear_model_for_normalization.pkl')

    print('plotting...')
    # plot data set
    plt.plot(years, avg_votes, 'ro')
    plt.xlabel('year')
    plt.ylabel('average votes')
    # plot the linear model
    y_pre = clf.predict(X)
    print(y_pre.shape)
    plt.plot(years, y_pre)
    plt.show()

def parse_json(filename):
    data = []
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data

main()
