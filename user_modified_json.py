import json
import csv
import sys,re

def user_modified_json():
    # Fixes ASCII error in Python csv writer module
    # reload(sys)
    # sys.setdefaultencoding('utf-8')

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
    for rows in data:
        row_data={}
        for keys in attribute_name:
            if not rows.has_key(keys):
                row_data[keys]=0
            else:
                if(keys==u'friends'):
                    row_data[keys]=len(rows[keys])
                if(keys==u'votes'):
                    row_data[keys]=rows[keys][u'funny']
                if(keys==u'compliments'):
                    row_data[keys]=len(rows[keys])
                if(keys==u'elite'):
                    row_data[keys]=len(rows[keys])
                if(keys=='yelping_since'):
                    since=[int(x.group()) for x in re.finditer(r'\d+', rows[keys])]
                    Yelp_time=1.0*(2016-since[0])+1.0*(5-since[1])/12                
                    row_data[keys]=Yelp_time
                if(keys==u'review_count'):
                    row_data[keys]=rows[keys]
                #if(keys==u'user_id'):
                #    row_data[keys]=rows[keys]
                if(keys==u'name'):
                    row_data[keys]=rows[keys]
                if(keys==u'type'):
                    row_data[keys]=rows[keys]
                if(keys==u'fans'):
                    row_data[keys]=rows[keys]
                if(keys==u'average_stars'):
                    row_data[keys]=rows[keys]                          
        array[rows[u'user_id']]=row_data
    return array
'''
with open('data.json', 'w') as outfile:
    json.dump(array, outfile)
'''
