# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 19:54:47 2016

@author: ZMP
"""
import numpy as np
import matplotlib.pyplot as plt
import heapq

numbins=50

def main():
    #load data for statistic analysis
    print('Loading dataset....')
    X,y=parse('./data_set/whole_set.csv')
    
    #Histogram data
    print('Generating Histogram...')
    hist, bins=np.histogram(y,bins=numbins)
    width = 0.7 * (bins[1] - bins[0])
    center=(bins[:-1] + bins[1:])/2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    
    #recalculate range to achieve uniformly distribution
    print('calculating threshhold...')
    thresharray=[]
    ylen=len(y)
    for i in reversed(range(9)):
        thresharray+=[nth_largest(int(ylen*1.0*(i+1)/10),y)]
    
    #reassign label
    print('reassigning label...')
    reassigned_y=[]
    for entry in y:
        reassigned_y+=[reassign_label(entry,thresharray,0)]
    
    #update dataset    
    print ('updating dataset')
    label=np.array([reassigned_y])
    data_set_updated=np.insert(X,[0],np.transpose(label),axis=1)
    print('writing to csv files...')
    np.savetxt('./data_set/classifications_dataset.csv', data_set_updated, delimiter=',')




def parse(filename):
    data_set = np.genfromtxt(filename, delimiter=',')
    X = data_set[:, 1:]
    y = data_set[:, 0]
    return X, y

def nth_largest(n, iter):
    return heapq.nlargest(n, iter)[-1]
    
def reassign_label(value, thresharray, i):
    if (i>8):
        return i
    if (value<thresharray[i]):
        return i
    return reassign_label(value, thresharray, i+1)
    
    
main()