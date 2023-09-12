"""
Image based video search engine: prototype
This file contains the ANNOY implementation for NNS
"""
import annoy
import numpy as np
import time


def annoy_build_tree(image_features, forest_size, metric, filename):
    """
    Builds the ANNOY tree with given features
    :param image_features: The n-dimensional feature vectors of the keyframes
    :param forest_size: The size of the forest
    :param metric: The distance metric ("angular", "euclidean", "manhattan", "hamming", or "dot")
    :param filename: The name of the save file without extension
    :return: A saved file containing the data of the tree
    """
    t1 = time.time()
    t = annoy.AnnoyIndex(len(image_features[0]), metric)  # Create the ANNOY index
    for i, feature in enumerate(image_features):  # Add the items one by one
        t.add_item(i, feature)
    t.build(forest_size)  # Build the tree
    t.save(rf'{filename}.ann')
    t2 = time.time()
    build_time = t2-t1
    return build_time


def annoy_search(image_features, metric, filename, k):
    """
    Searches the ANNOY tree for the closest neighbor to the query image
    :param image_features: The n-dimensional feature vectors of the query images
    :param metric: The distance metric ("angular", "euclidean", "manhattan", "hamming", or "dot")
    :param filename: The name of the save file without extension
    :param k: The amount of returned nearest neighbours
    :return: The k closest neighbors and their distance
    """
    t1 = time.time()

    # Set values
    queries, dim = image_features.shape
    idx = np.zeros((queries, k), dtype=np.int64)
    dist = np.zeros((queries, k), dtype=np.single)

    u = annoy.AnnoyIndex(dim, metric)  # Create an ANNOY index
    u.load(rf'{filename}.ann')  # Load the saved tree into the index

    # Find the k nearest  neighbours for every query
    for i, test_feature in enumerate(image_features):
        idx[i,:], dist[i,:] = u.get_nns_by_vector(test_feature, k, include_distances=True)
    t2 = time.time()
    time_per_query = (t2-t1)/queries
    return idx, dist, time_per_query
