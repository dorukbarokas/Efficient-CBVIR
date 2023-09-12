"""
Image based video search engine: prototype
This file contains the linear python implementation for NNS
"""
import time
import numpy as np


def matching_L2(K, frame_features, image_features):
    """
    Linear NNS
    :param K: the amount of nearest neighbors
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param image_features: The n-dimensional feature vectors of the query images
    :return: indices for the mAP calculation and time per query
    """
    t1 = time.time()
    num_test, _ = image_features.shape
    idx = np.zeros((num_test, K), dtype=np.int64)
    distances = np.zeros((num_test, K), dtype=np.single)
    # for each query determine the distance between all frames
    for row in range(num_test):
        query = image_features[row, :]
        dist = np.linalg.norm(query-frame_features, axis=1)  # calculate distances
        idx[row, :] = np.argsort(dist)[:K]  # sort based on indices
        distances[row, :] = dist[idx[row, :]]

    t2 = time.time()
    time_per_query = (t2-t1)/num_test
    return idx, distances, time_per_query
