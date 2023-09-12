"""
Image based video search engine: prototype
This file contains the HNSW implementation for NNS
"""
import time
import hnswlib


def hnsw_add(image_feature, max_elements, filename="hnswresult", space='l2', ef=10, ef_const=10, init=True, M = 16):
    """
    Partitioning of the data for HNSW
    :param image_feature: The n-dimensional feature vectors of the keyframes
    :param max_elements: Max allowed elements in graph, should be close to total amount of elements
    :param filename: Save file name
    :param space: The space metric (l2, cosine or ip)
    :param ef: Query time accuracy speed trade-off: higher ef leads to better accuracy but slower search
    :param ef_const: Parameter that controls speed/accuracy trade-off during the index construction.
    :param init: Indicates if a new graph should be build
    :param M: The maximum amount of connections per node in the graph
    :return: Save file with encoded graph
    """
    t1 = time.time()
    _, dim = image_feature.shape
    p = hnswlib.Index(space=space, dim=dim)  # Create a HNSW index
    if init:  # create a new graph or load an existing one
        p.init_index(max_elements=max_elements, ef_construction=ef_const, M=M)
    else:
        p.load_index(f"{filename}.bin", max_elements=max_elements)

    #  Build the graph
    p.set_ef(ef)
    p.add_items(image_feature)
    p.save_index(f"{filename}.bin")
    t2 = time.time()
    build_time = t2 - t1
    return build_time


def hnsw_search(image_features, k, filename="hnswresult", space='l2', ef=10):
    """
    Searche the HNSW graph
    :param image_features: The n-dimensional feature vectors of the query images
    :param k: Amount of Nearest Neighbours
    :param filename: Save file name
    :param space: The space metric (l2, cosine or ip)
    :param ef: Query time accuracy speed trade-off: higher ef leads to better accuracy but slower search
    :return: k closest neighbours and their distances
    """
    t1 = time.time()

    num_test, dim = image_features.shape
    p = hnswlib.Index(space=space, dim=dim)  # Create a new index
    p.load_index(f"{filename}.bin")  # Load the stored data into the index
    p.set_ef(ef)
    idx, distances = p.knn_query(image_features, k)  # Find the k nearest neighbours

    t2 = time.time()
    time_per_query = (t2 - t1) / num_test
    return idx, distances, time_per_query
