import json
import re

def preprocess():

    # read the review data set
    reviews = []
    with open('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json') as review_json_file:
        for line in review_json_file:
            reviews.append(json.loads(line))

    # read the user data set
    users = []
    with open('./yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json') as user_json_file:
        for line in user_json_file:
            users.append(json.loads(line))

    # read the taste words
    taste_words = []
    with open('./taste_words.txt') as taste_words_file:
        for line in taste_words_file:
            taste_words.append(line[:-1]) # ignore the \r

    # meta info
    print 'number of reviews:', len(reviews) # 2225213
    print 'number of users:', len(users) # 552339
    print 'number of taste words:', len(taste_words)

    # extract following features from reviews and users
    #   label: the quality of the review (i.e. the number of useful votes): int
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
    data_set = extract(reviews, users, taste_words)

    # write the data set into csv file
    with open('./data_set/data_set.csv', 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for data in data_set:
            csv_writer.writerow(data)

def extract(reviews, users, taste_words):
    data_set = []
    for review in reviews:
        user = findUser(review['user_id'], users)
        if not user:
            print 'did not find user:', review['user_id']
            continue
        data_set.append([review['votes']['useful'], len(review['text']), hasTaste(review['text'], taste_words), hasCost(review['text']), user['review_count'], sum(value for key, value in user['votes'].items()), len(user['elite']) > 0, sum(value for key, value in user['compliments'].items()), user['fans'], len(user['friends'])])
    return data_set

def findUser(user_id, users):
    for user in users:
        if user['user_id'] == user_id:
            return user
    return None

def hasTaste(text, taste_words):
    # convert to lower case
    text = text.lower()
    # find word in taste words list
    for word in taste_words:
        if text.find(word):
            return True
    return False

def hasCost(text):
    # check whether there is a number
    return re.match(r'\d', text) != None

preprocess()
