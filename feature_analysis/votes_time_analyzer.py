# Extract data from reviews and analyze the votes-time relationship

import json
import matplotlib.pyplot as plt
import datetime

def main():

    # load reviews data
    print('parsing...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')

    # extract duration of post date and votes count
    print('extracting...')
    vote_n_count = []
    for i in range(0, len(reviews), 500):
        review = reviews[i]
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        index = review_date.year - 2000
        while len(vote_n_count) - 1 < index:
            vote_n_count.append({'votes': 0, 'count': 0})
        # vote_n_count[index]['votes'] += sum(review['votes'].values())
        vote_n_count[index]['votes'] += int(review['votes']['useful'])
        vote_n_count[index]['count'] += 1

    # average the votes for each year
    avg_votes = []
    for item in vote_n_count:
        if item['count'] == 0:
            avg_votes.append(0)
        else:
            avg_votes.append(item['votes'] / item['count'])

    # prepare data to plot
    print('plotting...')
    plt.plot(range(2000, 2000+len(avg_votes)), avg_votes, 'ro')
    plt.show()

def parse_json(filename):
    data = []
    with open(filename) as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data

main()
