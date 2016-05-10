import json

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

    # meta info
    print 'number of reviews:', len(reviews) # 2225213
    print 'number of users:', len(users) # 552339

    # extract following features from reviews and users
    #   1. length of review
    #   2. has mentioned taste?
    #   3. has mentioned cost?
    #   4. has mentioned service? (undefined)
    #   5. count of reviews of the publisher
    #   6. count of votes of the publisher
    #   7. is the publisher an elite?
    #   8. count of compliment of the publisher
    #   9. count of fans of the publisher
    #   10. count of friends of the publisher
    #   label: the quality of the review (i.e. the number of useful votes)
    # note: the publisher means the user who publish the review
    data_set = extract(reviews, users)

    # write the data set into csv file
    with open('./data_set/data_set.csv', 'w+') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        for data in data_set:
            csv_writer.writerow(data)

def extract(reviews, users):
    pass

preprocess()
