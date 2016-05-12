# extract features from the yelp raw data set

from random import shuffle
import json
import re
import csv
import cPickle
import user_modified_json as uj
from sklearn import datasets, linear_model

def preprocess():

    # read the review data set
    reviews = []
    with open('yelp_academic_dataset_review.json') as review_json_file:
        for line in review_json_file:
            reviews.append(json.loads(line))
    
    # read the user data set
    users = uj.user_modified_json()

    # read the votes_time_lr prediction model
    with open('votes_time_lr.pkl', 'rb') as fid:
        lr = cPickle.load(fid)
    
    # read the taste words
    taste_words = []
    with open('taste_words.txt') as taste_words_file:
        for line in taste_words_file:
            taste_words.append(line[:-1]) # ignore the \r

    # extract following features from reviews and users
    #   label: the quality of the review
    #   1. length of review: int
    #   2. has mentioned taste?: boolean
    #   3. has mentioned cost?: boolean
    #   4. count of reviews of the publisher: int
    #   5. count of votes of the publisher: int
    #   6. is the publisher an elite?: boolean
    #   7. count of compliment of the publisher: int
    #   8. count of fans of the publisher: int
    #   9. count of friends of the publisher: int
    # note: the publisher means the user who publish the review
    # note2: the label is placed at the 0 index
    myRange =  range(0,500000) 
    shuffle(myRange)
    rand_smpl = [ reviews[i] for i in sorted(myRange[:100000]) ]
    data_set = extract(rand_smpl, users, taste_words,lr)

    # write the data set into csv file
    with open('data_set.csv', 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerows(data_set)
    
    myRange =  range(0,500000) 
    shuffle(myRange)
    rand_smpl = [ reviews[i] for i in sorted(myRange[:100000]) ]
    data_set = extract(rand_smpl, users, taste_words,lr)

    # write the data set into csv file
    with open('data_set_test.csv', 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerows(data_set)

def extract(reviews, users, taste_words, lr):
    data_set = []
    for review in reviews:
        user = findUser(review['user_id'], users)
        if not user:
            print 'did not find user:', review['user_id']
            continue
        post_time=[int(x.group()) for x in re.finditer(r'\d+', review['date'])]
        post_duration=1.0*(2016-post_time[0])+1.0*(5-post_time[1])/12+1.0*(1.0-post_time[2])/12/30
        factor=lr.predict([post_duration])[0]   
        review_quality=1.0*review['votes']['useful']/factor
        if(review_quality>0):
            data_set.append([review_quality, len(review['text']), hasTaste(review['text'], taste_words), hasCost(review['text']), user['review_count'], user['votes'], user['elite'], user['compliments'], user['fans'], user['friends']])
    return data_set

def findUser(user_id, users):
    if users.has_key(user_id):
        return users[user_id]
    return None

def hasTaste(text, taste_words):
    # convert to lower case
    text = text.lower()
    # find word in taste words list
    for word in taste_words:
        if text.find(word) != -1:
            return True
    return False

def hasCost(text):
    # check whether there is a number
    return re.match(r'\d', text) != None

preprocess()
