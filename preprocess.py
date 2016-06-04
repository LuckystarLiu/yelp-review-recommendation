# extract features from the yelp raw data set reviews and users

import random
import json
import csv
import datetime
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import cPickle
# import user_modified_json as uj
# from sklearn import datasets, linear_model

def main():

    # read the review data set
    print('parsing review data...')
    reviews = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json')
    
    # read the user data set
    print('parsing user data...')
    users = parse_json('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json')

    # NOTE: use count of votes as label first
    # read the votes_time_lr prediction model
    # with open('votes_time_lr.pkl', 'rb') as fid:
    #     lr = cPickle.load(fid)
    
    # sample the data
    print('sampling from '+str(len(reviews))+' data set...')
    review_samples = sample(reviews, 100000)

    # extract
    print('extracting features...')
    data_set = extract(review_samples, users)

    # split into training and testing data set
    print('splitting data set...')
    portion = 0.6
    train_len = int(len(data_set)*portion)
    train_set = data_set[:train_len]
    test_set = data_set[train_len:]

    # write the data set into csv file
    print('writing training data set...')
    write_to_csv('./data_set/train_set.csv', train_set)
    print('writing testing data set...')
    write_to_csv('./data_set/test_set.csv', test_set)

def extract(reviews, users):

    # convert to dictionary and store only useful info for fast reading
    users = extract_users(users)

    # get bag of words for all reviews
    bag_of_words = get_bag_of_words([review['text'] for review in reviews])

    # extract features
    data_set = []
    now = datetime.datetime.now()
    for i in range(len(reviews)):

        review = reviews[i]

        # check user
        if review['user_id'] not in users:
            print('did not find user:', review['user_id'])
            continue

        # find the user
        user = users[review['user_id']]

        # normalize: review_quality = votes/day
        review_date = datetime.datetime.strptime(review['date'], '%Y-%m-%d')
        delta = now - review_date
        days = divmod(delta.total_seconds(), 24 * 60 * 60)[0]
        review_quality = sum(review['votes'].values()) / days

        # combine the features
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
                         len(review['text'])]
                         + list(bag_of_words[i]))
        
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

# refer to http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
def get_bag_of_words(texts):

    # count the occurrence of words in all texts
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(texts)
    X_train_counts = X_train_counts.toarray()

    # transform from occurrence to frequency
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf.toarray()

def write_to_csv(filename, data_set):
    with open(filename, 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerows(data_set)

main()
