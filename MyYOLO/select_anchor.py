import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

with open('./data/car/image/train_1w.csv', 'r', encoding='utf-8') as src_file:
    src_file.readline()
    with open('./data/car/image/kmeans_wh.csv', 'w') as kfile:
        for line in src_file:
            line = line.strip('\n')
            label_list = line.split(',')
            image_name = label_list[0]
            position_list = list(map(lambda x : x.split('_'), label_list[1].split(';')))

            for position in position_list:
                if position[0] == '':
                    continue

                kfile.write(str(position[2]) + ',' + str(position[3]) + '\n')


df = pd.read_csv('./data/car/image/kmeans_wh.csv', sep=',', header=None)
data = df.values

km = KMeans(n_clusters=9, init='k-means++', random_state=45)
km.fit(data)

print(km.cluster_centers_)