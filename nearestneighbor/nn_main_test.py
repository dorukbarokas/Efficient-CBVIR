"""
Image based video search engine: prototype
This file contains the testing environment for the module.
"""
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('nearestneighbor/nn_main_test.py')))

import nn_annoy
import nn_hnsw
import nn_linear
import nn_faiss


def nns(frame_features_in, image_features_in, method, k = None, k_percentage = None, annoy_forest_size=10,
        annoy_metric='euclidean', hnsw_batch_size=0.10,
        ef_const = 100, frame_labels = None, image_labels = None, build=True, hnsw_m = 8):
    """
    Test function for different nns methods and arguments
    :param frame_features: The n-dimensional feature vectors of the keyframes
    :param image_features: The n-dimensional feature vectors of the query images
    :param method: The method to be used
    :param k: A fixed number of nearest neighbours to be found
    :param k_percentage: The percentage of nearest neighbours to be found
    :param annoy_forest_size: Param forest size for ANNOY
    :param annoy_metric: Distance metric for ANNOY
    :param hnsw_batch_size: Batch size in percentage for hnsw
    :param ef_const: Param ef_const for HNSW
    :param frame_labels: The labels of the feature vectors of the keyframes
    :param image_labels: The labels of the feature vectors of the query images
    :param build: Indicates if data needs to be partitioned
    :param hnsw_m: Param M for HNSW
    :return: The build_time, time_per_query, total_time, mAP and recall
    """
    build_time = 0
    mAP = None
    recall = None
    n_frames, _ = frame_features_in.shape

    if k is None and k_percentage is None:
        raise("Specify one form of k")

    if k_percentage is not None:
        k = max(1, round(k_percentage/100 * n_frames))

    # normalization
    eftrain_norm = np.linalg.norm(frame_features_in, axis=1)
    eftrain_norm = np.expand_dims(eftrain_norm, axis=1)
    eftest_norm = np.linalg.norm(image_features_in, axis=1)
    eftest_norm = np.expand_dims(eftest_norm, axis=1)
    frame_features_in = frame_features_in / eftrain_norm
    image_features_in = image_features_in / eftest_norm


    if method.lower() == 'annoy':  # NNS by using annoy
        if build:  # Build the network
            build_time = nn_annoy.annoy_build_tree(frame_features_in, annoy_forest_size, annoy_metric, 'AnnoyTree')
        # Search the network
        nns_res, dist, time_per_query = nn_annoy.annoy_search(image_features_in, annoy_metric, 'AnnoyTree', k)

    elif method.lower() == 'hnsw':  # NNS by using HNSW
        if build:  # Build the network
            build_time = nn_hnsw.hnsw_add(frame_features_in, max_elements=len(frame_features_in), ef_const=ef_const, M=hnsw_m)
        # Search the network
        nns_res, dist, time_per_query = nn_hnsw.hnsw_search(image_features_in, k, ef = k)

    elif method.lower() == 'hnsw_batch':  # NNS by using HNSW and building the graph with batches
        init = True
        if build:  # Build the network in batches of given size
            batch_time = []
            hnsw_batch_size = int(hnsw_batch_size*n_frames)
            for index in range(0, len(frame_features_in), hnsw_batch_size):
                frame_feature = frame_features_in[index:index+hnsw_batch_size]
                build_time = nn_hnsw.hnsw_add(frame_feature, max_elements=index + hnsw_batch_size, init=init, ef_const=100)
                batch_time.append(build_time)
                init = False
            print(f"Total Build time: {np.sum(batch_time)}\nMean build time: {np.mean(batch_time)}")
        # Search the network
        nns_res, dist, time_per_query = nn_hnsw.hnsw_search(image_features_in, k, ef=k)

    elif method.lower() == 'linear':
        build_time = 0  # No build time since there is no network, just comparisons
        # Linear search
        nns_res, dist, time_per_query = nn_linear.matching_L2(k, frame_features_in, image_features_in)

    # Several faiss implementations
    elif method.lower() == 'faiss_flat_cpu':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_flat(image_features_in, frame_features_in, k, False)
    elif method.lower() == 'faiss_flat_gpu':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_flat(image_features_in, frame_features_in, k, True)

    elif method.lower() == 'faiss_hnsw':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_hnsw(image_features_in, frame_features_in, k)

    elif method.lower() == 'faiss_pq':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_pq(image_features_in, frame_features_in, k)

    elif method.lower() == 'faiss_lsh':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_lsh(image_features_in, frame_features_in, k)

    elif method.lower() == 'faiss_ivf_cpu':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_ivf(image_features_in, frame_features_in, k, False)
    elif method.lower() == 'faiss_ivf_gpu':
        nns_res, dist, build_time, time_per_query = nn_faiss.faiss_ivf(image_features_in, frame_features_in, k, True)

    else:
        raise Exception("No method available with that name")

    total_time = build_time + time_per_query
    if (frame_labels is not None) and (image_labels is not None):  # If labels are given calculate mAP and recall
        mAP = cal_mAP(nns_res, frame_labels, image_labels)
        recall = cal_recall(nns_res, frame_labels, image_labels)
        print(f"maP:\t\t\t\t{mAP}")
        print(f"Recall:\t\t\t{recall}")

    print(f"Build time:\t\t\t{build_time}")
    print(f"Search time per query:\t{time_per_query}")
    print(f"&{np.round((build_time+(time_per_query*args.n_queries))*1000,2)}\t&{np.round(build_time*1000,2)}\t&{np.round(time_per_query*1000,2)}\t&{np.round(mAP,2)}\t&{np.round(recall,2)}")

    return build_time, time_per_query, total_time, mAP, recall


def cal_mAP(idx, labels_train, labels_test):
    """
    Calculate the mean average precision
    :param idx: The indices of the NN keyframes
    :param labels_train: The labels of the keyframes
    :param labels_test: The labels of the query images
    :return: The mAP
    """
    # Code provided by current research group
    num_queries, K = idx.shape
    matched = np.zeros_like(idx, dtype=np.int16)
    for i in range(num_queries):
        count = 0
        for j in range(K):
            if labels_test[i] == labels_train[idx[i, j]]:
                count += 1
                matched[i, j] = count
    AP = np.sum(matched/(np.array(range(K))+1)/K, axis=1)
    mAP = AP.mean()
    return mAP


def cal_recall(idx, labels_train, labels_test):
    """
    Calculate the recall
    :param idx: The indices of the NN keyframes
    :param labels_train: The labels of the keyframes
    :param labels_test: The labels of the query images
    :return: The mean of the recall of the queries
    """
    num_queries, K = idx.shape
    recall = []
    for i in range(num_queries):
        count = 0
        for j in range(K):
            if labels_test[i] == labels_train[idx[i, j]]:
                count += 1
        total_labels = len(np.where(labels_train == labels_test[i])[0])
        recall.append((count/total_labels) if total_labels != 0 else 0)
    mrecall = np.array(recall).mean()
    return mrecall


def cal_precision(idx, labels_train, labels_test):
    """
    Calculate the precision
    :param idx: The indices of the NN keyframes
    :param labels_train: The labels of the keyframes
    :param labels_test: The labels of the query images
    :return: The mean of the precision of the queries
    """
    num_queries, K = idx.shape
    precision = []
    for i in range(num_queries):
        count = 0
        for j in range(K):
            if labels_test[i] == labels_train[idx[i, j]]:
                count += 1
        precision.append(count/K)
    mprecision = np.array(precision).mean()
    return mprecision


def main():
    """
    Main body to test the the different implementations via calling the file in terminal
    :return:
    """
    # Load data
    frame_features = np.load((os.path.abspath(r'data/frames.npy')))
    frame_labels = np.load((os.path.abspath(r'data/frames_labels.npy')))
    image_features = np.load((os.path.abspath(r'data/images.npy')))
    image_labels = np.load((os.path.abspath(r'data/images_labels.npy')))

    # Create smaller dataset with equal distribution
    # params to be set dependent on dataset (this case CIFAR-10)
    np.random.seed(1234) # for generating consistent data
    if args.n_frames < 50000:
        dataset_new_size = args.n_frames
        dataset_n_classes = 10
        new_dataset_indices = []
        assert dataset_new_size % dataset_n_classes == 0  # Check if data can be equally separated

        # For each class generate a random subset
        for i in range(dataset_n_classes):
            new_dataset_indices.append(
                np.random.choice(np.where(frame_labels == i)[0], int(dataset_new_size / dataset_n_classes), replace=False))

        # Combine the random data
        new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
        frame_features = frame_features[new_dataset_indices_flatten]
        frame_labels = frame_labels[new_dataset_indices_flatten]

    # select a random number of queries
    if args.n_queries < 10000:
        r_queries = np.random.choice(len(image_features), args.n_queries, False)
        image_features = image_features[r_queries]
        image_labels = image_labels[r_queries]

    nns(frame_features, image_features, args.method, k_percentage=args.k_percentage, frame_labels=frame_labels, image_labels=image_labels)


if __name__ == "__main__":
    # definition of parser arguments
    a = argparse.ArgumentParser()
    a.add_argument("--n_frames", default=50000, type=int, help="amount of input frames")
    a.add_argument("--n_queries", default=1, type=int, help="amount of queries")
    a.add_argument("--method", default="linear", type=str, help="The NNS method to be used")
    a.add_argument("--k", default=None, type=int, help="The amount of nearest neighbors")
    a.add_argument("--k_percentage", default=None, type=int, help="The amount of nearest neighbors %")

    # more for testing purposes
    a.add_argument("--forest_size", default=10, type=int)
    a.add_argument("--annoy_metric", default="angular", type=str)
    a.add_argument("--batch_size", default=500, type=int)
    args = a.parse_args()
    main()

