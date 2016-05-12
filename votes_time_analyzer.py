# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:01:28 2016
Extract data from json reviews and analyze the votes-time relationship
@author: ZMP
"""

import json
import matplotlib.pyplot as plt
import sys,re
reload(sys)
sys.setdefaultencoding('utf-8')
data = []
attribute_name={}
f=open('yelp_academic_dataset_review.json')
i=0   
likes_time=[]  #[[num_of_votes, "2016-05" - post_date]]
pd=[]  #["2016-05" - post_date]
steps=20

# 1.Loads the review json file, find the votes and post date of each review
for line in f:
    data.append(json.loads(line))
for rows in data:
    post_time=[int(x.group()) for x in re.finditer(r'\d+', rows[u'date'])]
    post_duration=1.0*(2016-post_time[0])+1.0*(5-post_time[1])/12
    likes_time+=[[rows[u'votes'][u'useful'],post_duration]]
    pd+=[post_duration]

# prepare data to plot
minpd=min(pd)
maxpd=max(pd)
dur_array=[0.0]*(steps+1)
dur_array_cnt=[0.0]*(steps+1)
for i in likes_time:
    ind=int(round((i[1]-minpd)/(maxpd-minpd)*steps))
    dur_array[ind]+=i[0]
    dur_array_cnt[ind]+=1
for i in range(steps+1):
    dur_array[i]/=dur_array_cnt[i]
time_x=[1.0/steps*i*(maxpd-minpd)+minpd for i in range(steps+1)]
plt.plot(time_x,dur_array)

