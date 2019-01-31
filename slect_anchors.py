#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:24:38 2019

@author: wxz
"""
import pandas as pd
from sklearn.cluster import KMeans

with open('/home/wxz/yolo/train_1w.csv', 'r') as src_file:
    with open('/home/wxz/yolo/kmeans_wh.csv', 'w') as kfile:
        for line in src_file:
            label_list = line.strip().split(',')
            image_name = label_list[0]
            position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))
            if position_list == [['']]:
                continue
            for position in position_list:
                

                kfile.write(str(position[2]) + ',' + str(position[3]) + '\n')


df = pd.read_csv('/home/wxz/yolo/kmeans_wh.csv', sep=',', header=None)
data = df.values

km = KMeans(n_clusters=15, init='k-means++', random_state=45)
km.fit(data)
print(km.cluster_centers_)

