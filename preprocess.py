# extract features from the yelp raw data set reviews and users

import random
import json
import csv
import datetime
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.externals import joblib

word_count = 100

def main():

    # read the review data set
    print('parsing review data...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')
    
    # read the user data set
    print('parsing user data...')
    users = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json')

    # use only the reviews after 2008
    valid_reviews = []
    for review in reviews:
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        if review_date.year < 2008: 
            continue
        valid_reviews.append(review)
    reviews = valid_reviews

    # sample the data
    sample_num = len(reviews)
    print('sampling...', sample_num, 'out of', len(reviews))
    review_samples = sample(reviews, sample_num)

    # convert to dictionary and store only useful info for fast reading
    print('converting users from list to dict...')
    users = extract_users(users)

    # extract
    print('extracting features...')
    data_set = extract(review_samples, users)

    # split into training and testing data set
    print('splitting data set...')
    portion = 0.7
    train_len = int(data_set.shape[0]*portion)
    train_set = data_set[:train_len]
    test_set = data_set[train_len:]

    # writing to csv file
    print('writing to csv files...')
    np.savetxt('./data_set/train_set.csv', train_set, delimiter=',')
    np.savetxt('./data_set/test_set.csv', test_set, delimiter=',')

def extract(reviews, users):

    # load the linear model for normalization
    clf = joblib.load('./normalization/linear_model_for_normalization.pkl')

    # get user info and length of review as text features
    print('extracting dense features...')
    data_set = []
    texts = []
    now = datetime.datetime.now()
    for review in reviews:

        # check user
        if review['user_id'] not in users:
            print('did not find user:', review['user_id'])
            continue

        # find the user
        user = users[review['user_id']]

        # normalize: review_quality = votes/day
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        normalizor = clf.predict(np.array([[review_date.year]]))[0][0]
        review_quality = sum(review['votes'].values()) / normalizor

        # put the data into data set
        data_set.append([review_quality,\
                         user['review_count'],\
                         user['average_stars'],\
                         user['vote_funny_count'],\
                         user['vote_useful_count'],\
                         user['vote_cool_count'],\
                         user['friend_count'],\
                         user['elite_count'],\
                         user['compliment_count'],\
                         user['fan_count'],\
                         len(review['text'])])

        # review texts for getting bag of words
        texts.append(review['text'])
        
    # get bag of words for all reviews as text features
    print('extracting bag of words...')
    bag_of_words = get_bag_of_words(texts)

    # combine explicit features and bag of words features
    print('combining dense features and bags of words...')
    data_set = np.hstack([data_set, bag_of_words])

    return data_set

# convert json to dictionary
def extract_users(users):
    users_dict = {}
    for user in users:
        user_dict = {}
        # extract useful infos
        user_dict['review_count'] = int(user['review_count'])
        user_dict['average_stars'] = float(user['average_stars'])
        user_dict['vote_funny_count'] = int(user['votes']['funny'])
        user_dict['vote_useful_count'] = int(user['votes']['useful'])
        user_dict['vote_cool_count'] = int(user['votes']['cool'])
        user_dict['friend_count'] = int(len(user['friends']))
        user_dict['elite_count'] = int(len(user['elite']))
        user_dict['compliment_count'] = sum(user['compliments'].values())
        user_dict['fan_count'] = int(user['fans'])
        # put in the users dictionary
        users_dict[user['user_id']] = user_dict
    return users_dict

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

# detail refer to http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
def get_bag_of_words(texts):

    # count the occurrence of words in all texts
    count_vect = CountVectorizer(max_features = 500)
    X_train_counts = count_vect.fit_transform(texts)

    # transform from occurrence to frequency
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # downsample (dimension reduction)
    print('downsampling bag of words...')
    pca = PCA(n_components = word_count)

    return pca.fit_transform(X_train_tfidf.toarray())

main()
