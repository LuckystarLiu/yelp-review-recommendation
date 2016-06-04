import json
import csv
import sys,re

def user_modified_json():

    # Load json files by lines into data
    data = []
    attribute_name={}
    user_file = open('yelp_academic_dataset_user.json')
    for line in user_file:
        data.append(json.loads(line))
            
    # extract csv headers, 10000 samples should be sufficient
    i=0
    for rows in data:
        if i>=10000:
            break;
        i+=1;
        for keys in rows:
            if not attribute_name.has_key(keys):
                attribute_name[keys]=0
    print attribute_name

    #Process data into csv compatible and desired format
    array={}
    for user in data:
        my_user={}
        for key in attribute_name:
            if not user.has_key(key):
                my_user[key]=0
            else:
                if(key=='friends'):
                    my_user[key]=len(user[key])
                if(key=='votes'):
                    my_user[key]=user[key][u'funny']
                if(key=='compliments'):
                    my_user[key]=len(user[key])
                if(key=='elite'):
                    my_user[key]=len(user[key])
                if(key=='yelping_since'):
                    since=[int(x.group()) for x in re.finditer(r'\d+', user[key])]
                    Yelp_time=1.0*(2016-since[0])+1.0*(5-since[1])/12                
                    my_user[key]=Yelp_time
                if(key=='review_count'):
                    my_user[key]=user[key]
                #if(key==u'user_id'):
                #    my_user[key]=user[key]
                if(key=='name'):
                    my_user[key]=user[key]
                if(key=='type'):
                    my_user[key]=user[key]
                if(key=='fans'):
                    my_user[key]=user[key]
                if(key=='average_stars'):
                    my_user[key]=user[key]                          
        array[user['user_id']]=my_user
    return array
'''
with open('data.json', 'w') as outfile:
    json.dump(array, outfile)
'''
