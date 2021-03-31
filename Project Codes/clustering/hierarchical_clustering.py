# GOKHAN HAS - 161044067
# DATA MINING PROJECT
# HIERARCHICAL CLUSTERING ALGORITHM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


# read data from csv
def read_data():
    data = pd.read_csv('fifa21.csv')
    arr = StandardScaler().fit_transform(data.iloc[:, [3, 6]].values)
    return arr


# küme listesi oluşturma
def get_cluster_data():
    data = read_data()
    clusters_list = list()
    for i in range(len(data)):
        clusters_list.append(data[i])
    for i in range(len(clusters_list)):
        clusters_list[i] = clusters_list[i].reshape(1, len(clusters_list[i]))
    return clusters_list


# İki veri arasındaki mesafe hesaplanır.
# Öklid yöntemi kullanılmıştır.
# Bu işlem uzun sürüyor.
def find_max_distance(data1, data2):
    max_distance = 0
    for i in range(len(data1)):
        for x in range(len(data2)):
                distance_ = distance.euclidean(data1[i], data2[x])
                if distance_ > max_distance:
                    max_distance = distance_
    return max_distance


# Veri kümesindeki noktalar arasındaki en kısa mesafeyi hesaplar.
def find_min_distance_clusters(data):
    cluster_ind1 = 0
    cluster_ind2 = 0
    min_of_max_distance = np.inf
    for i in range(len(data)):
        for x in range(len(data)):
            if i == x or x < i:
                continue
            else:
                max_dis = find_max_distance(data[i], data[x])
                if max_dis < min_of_max_distance:
                    min_of_max_distance = max_dis
                    cluster_ind1 = i
                    cluster_ind2 = x
    return cluster_ind1, cluster_ind2


# Aynı kümeden çıkarılan noktalar, diğer kümelerden silinirken belirtilen kümeye eklenir.
def merging_cluster_data(data_new, ind_cluster1, ind_cluster2):
    temp = np.concatenate((data_new[ind_cluster1], data_new[ind_cluster2]), axis=0)
    if ind_cluster1 > ind_cluster2:
        del data_new[ind_cluster1]
        del data_new[ind_cluster2]
    else:
        del data_new[ind_cluster2]
        del data_new[ind_cluster1]
    data_new.append(temp)
    return data_new

# Kümeleri bulmak noktalar arasında yapılır.
def main(num_clusters):
    data = get_cluster_data()
    while len(data) > num_clusters:
        first, second = find_min_distance_clusters(data)
        data_2 = merging_cluster_data(data, first, second)
        data = data_2
    return data


if __name__ == '__main__':
    cluster_num = 5
    list_of_clusters = main(cluster_num)
    for i in range(len(list_of_clusters)):
        plt.scatter(list_of_clusters[i][:, 0], list_of_clusters[i][:, 1])
    plt.title("Hierarchical Clustering Algorithm, Cluster Number : {}".format(cluster_num))
    plt.show()
