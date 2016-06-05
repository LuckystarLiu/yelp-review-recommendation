# -*- coding: utf-8 -*-
"""
Created on Wed May 11 01:47:06 2016

Extract data from json reviews and analyze the votes-time relationship
Perform regression on the first 2 year data, more detailed
Should run after votes_time_analyzer.py

@author: ZMP
"""

 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
month_r=[minpd,minpd+2.0]#range 1
day_r=[minpd,minpd+2.0/6.0]#range 2
steps=30 #plotting steps
threshold=0 #num_of_votes threshold

pd_m=[]
likes_time_m=[]
pd_d=[]
likes_time_d=[]

#extracting data (first 2 years/first 4 months) from the 12 years data
for i in pd:
    if i>=month_r[0] and i<=month_r[1]:
        pd_m+=[i]
    if i>=day_r[0] and i<=day_r[1]:
        pd_d+=[i]
for i in likes_time:
    if i[1]>=month_r[0] and i[1]<=month_r[1]:
        likes_time_m+=[i]
    if i[1]>=day_r[0] and i[1]<=day_r[1]:
        likes_time_d+=[i]
minpd_m=min(pd_m)
maxpd_m=max(pd_m)
minpd_d=min(pd_d)
maxpd_d=max(pd_d)

#preparation of data_to_plot (2 years)
dur_array_m=[0.0]*(steps+1)
dur_array_cnt_m=[0.0]*(steps+1)
for i in likes_time_m:
    if i[0]>=threshold:
        ind=int(round((i[1]-minpd_m)/(maxpd_m-minpd_m)*steps))
        dur_array_m[ind]+=i[0]
        dur_array_cnt_m[ind]+=1
for i in range(steps+1):
    if dur_array_cnt_m[i]==0:
        dur_array_m[i]=dur_array_m[i-1]
    else:
        dur_array_m[i]/=dur_array_cnt_m[i]
time_x_m=[1.0/steps*i*(maxpd_m-minpd_m)+minpd_m for i in range(steps+1)]

#preparation of data_to_plot (4 months)
dur_array_d=[0.0]*(steps+1)
dur_array_cnt_d=[0.0]*(steps+1)
for i in likes_time_d:
    if i[0]>=threshold:
        ind=int(round((i[1]-minpd_d)/(maxpd_d-minpd_d)*steps))
        dur_array_d[ind]+=i[0]
        dur_array_cnt_d[ind]+=1
for i in range(steps+1):
    if dur_array_cnt_d[i]==0:
        dur_array_d[i]=dur_array_d[i-1]
    else:
        dur_array_d[i]/=dur_array_cnt_d[i]
time_x_d=[1.0/steps*i*(maxpd_d-minpd_d)+minpd_d for i in range(steps+1)]

#plotting the votes count mean against time (2-year span)
plt.title('votes count mean against time (2-year span)')
plt.xlabel('time/year')
plt.ylabel('average num of votes')
plt.plot(time_x_m,dur_array_m)

#plotting the votes count mean against time (12 year span)
plt.figure()
plt.title('votes count mean against time (4-month span)')
plt.xlabel('time/year')
plt.ylabel('average num of votes')
plt.plot(time_x_d,dur_array_d)

#plotting the votes count mean against time (12-year span)
plt.figure()
plt.title('votes count mean against time (12-year span)')
plt.xlabel('time/year')
plt.ylabel('average num of votes')
plt.plot(time_x,dur_array)

#linear regression of votes count mean against time (2-year span)
plt.figure()
plt.title('linear regression of votes count mean against time (2-year span)')
plt.xlabel('time/year')
plt.ylabel('average num of votes')
plt.scatter(time_x_m,dur_array_m)
lr=linear_model.LinearRegression()
print np.array(time_x_m).T
lr.fit(np.array(time_x_m).reshape((len(time_x_m),1)),np.array(dur_array_m))
print('Coefficients: ', lr.coef_)
plt.plot(time_x_m,lr.predict(np.array(time_x_m).reshape((len(time_x_m),1))),'r')
plt.legend([ 'LR prediction','origin data scatter'])
