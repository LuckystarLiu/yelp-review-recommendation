# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:01:28 2016
Extract data from json reviews and analyze the votes-time relationship
@author: ZMP
"""

import json
import matplotlib.pyplot as plt
import datetime

def main():

    # load reviews data
    print('parsing...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')

    # extract duration of post date and votes count
    print('extracting...')
    votes = []
    days = []
    special_date = datetime.datetime(1970, 1, 1)
    for i in range(0, len(reviews), 500):
        review = reviews[i]
        votes_count = sum(review['votes'].values())
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        delta = review_date - special_date
        day = divmod(delta.total_seconds(), 24 * 60 * 60)[0]
        votes.append(votes_count)
        days.append(day)

    # prepare data to plot
    print('plotting...')
    plt.plot(days, votes, 'ro')
    plt.show()

def parse_json(filename):
    data = []
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data

main()
