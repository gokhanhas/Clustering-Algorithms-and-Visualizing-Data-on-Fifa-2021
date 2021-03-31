# GOKHAN HAS - 161044067
# DATA MINING PROJECT
# MEAN SHIFT ALGORITHM

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# read data from csv
def read_data():
    data = pd.read_csv('fifa21.csv')
    arr = StandardScaler().fit_transform(data.iloc[:, [3, 6]].values)
    return arr


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def gaussian_distance(distance_x, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (distance_x / bandwidth) ** 2)


class MeanShift(object):
    def __init__(self, distance=gaussian_distance):
        self.distance = distance

    def shifter_algorithm(self, points, radius):
        shift_points = np.array(points)
        shifting = [True] * points.shape[0]

        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self.point_shift(shift_points[i], points, radius)
                dist = distance(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > 1e-4

            if max_dist < 1e-4:
                break

        cluster_ids = self.get_cluster(shift_points.tolist())
        return shift_points, cluster_ids

    def point_shift(self, point, points, radius):
        shift_x = 0.0
        shift_y = 0.0
        scale = 0.0
        for p in points:
            dist = distance(point, p)
            weight = self.distance(dist, radius)
            shift_x += p[0] * weight
            shift_y += p[1] * weight
            scale += weight
        shift_x = shift_x / scale
        shift_y = shift_y / scale
        return [shift_x, shift_y]

    def get_cluster(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if len(cluster_ids) == 0:
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = distance(point, center)

                    if dist < 1e-1:
                        cluster_ids.append(cluster_centers.index(center))

                if len(cluster_ids) < i + 1:
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids


def main():
    x = read_data()
    kernel_value = 0.3
    mean_shifter = MeanShift()
    y, mean_shift_result = mean_shifter.shifter_algorithm(x, kernel_value)

    np.set_printoptions(precision=3)

    #print('input: {}'.format(x))
    #print('assined clusters: {}'.format(mean_shift_result))

    cluster_number = max(mean_shift_result) + 1

    color = []
    for i in range(np.unique(mean_shift_result).size):
        color.append((random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))

    for i in range(len(mean_shift_result)):
        plt.scatter(x[i, 0], x[i, 1], color=color[mean_shift_result[i]])

    plt.title('Mean Shift Clustering, Kernel Value : {}, Cluster Number : {}'.format(kernel_value, cluster_number))
    plt.show()


if __name__ == '__main__':
    main()
