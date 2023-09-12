"""
Image based video search engine: prototype
This file contains functions to determine the amount of NN to be found and evaluation of the selector
"""
import os
import pandas as pd
import numpy as np
import argparse

import nn_main_test
from main import method_selector


def main():
    """
    Call the different function to collect data on k_percentage
    :return: save files with the stored data
    """
    methods = ['linear', 'faiss_flat_cpu', 'faiss_flat_gpu', 'faiss_hnsw', 'faiss_lsh']

    timevsk(methods, 25, 270, 'timevsk270')
    timevsk(methods, 25, 8100, 'timevsk8100')
    timevsk(methods, 25, 50000, 'timevsk50000')
    timevsk(methods, 25, 10, 'timevsk10')
    timevsk(methods, 25, 20, 'timevsk20')
    timevsk(methods, 25, 30, 'timevsk30')
    timevsk(methods, 25, 40, 'timevsk40')
    timevsk(methods, 25, 50, 'timevsk50')


def to_csv(data_names, data, filename, mode='w'):
    """
    put data in a csv
    :param data_names: the names of the different data in the data list (["name1","name2"])
    :param data: data for to be stored ([[1,2,3],[4,5,6]])
    :param filename: the name of the savefile
    :param mode: if a file should be written or appended
    :return: a csv with the data
    """
    d = dict()
    if isinstance(data, list):
        for name, item in zip(data_names, data):
            d[name] = item
        length = len(data[0])
    else:
        d[data_names] = data
        length = len(data)

    df = pd.DataFrame(data=d, index=range(length))
    df.to_csv(f'test_data/{filename}.csv', index=True, header=True, mode=mode)


def timevsk(methods, max_percentage, size, filename):
    """
    Collect data for each method for the percentage of nearest neighbors to be taken
    :param methods: the methods to evaluate
    :param max_percentage: the max_percentage to take for k
    :param size: the size of the sub-dataset to be made (multiple of 10 for CIFAR-10)
    :param filename: The name of the save file
    :return: A csv with all the collected data
    """
    n_queries = 100
    # Create smaller dataset with equal distribution
    np.random.seed(1234)  # for generating consistent data
    # params to be set dependend on dataset (this case CIFAR-10)
    if size < 50000:
        dataset_n_classes = 10
        new_dataset_indices = []
        assert size % dataset_n_classes == 0

        for i in range(dataset_n_classes):
            new_dataset_indices.append(
                np.random.choice(np.where(frame_labels == i)[0], int(size / dataset_n_classes),
                                 replace=False))

        new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
        frame_features_subset = frame_features[new_dataset_indices_flatten]
        frame_labels_subset = frame_labels[new_dataset_indices_flatten]
    else:
        frame_features_subset = frame_features
        frame_labels_subset = frame_labels

    r_queries = np.random.choice(len(image_features), n_queries, False)
    image_features_subset = image_features[r_queries]
    image_labels_subset = image_labels[r_queries]

    search_arr = []
    mAP_arr = []
    recall_arr = []
    method_arr = []
    for method in methods:
        init = True
        for i in np.linspace(0.1, max_percentage,
                             50):  # switch out for np.linspace(0.1, max_percentage, 25)? (50...50000)
            k_float = (i / 100) * len(frame_features_subset)
            k = max(1, round(k_float))
            if init:
                build_time, search, _, mAP, recall = nn_main_test.nns(frame_features_subset, image_features_subset,
                                                                      method, k,
                                                                      frame_labels=frame_labels_subset,
                                                                      image_labels=image_labels_subset, build=True)
                init = False
            else:
                build_time, search, _, mAP, recall = nn_main_test.nns(frame_features_subset, image_features_subset,
                                                                      method, k,
                                                                      frame_labels=frame_labels_subset,
                                                                      image_labels=image_labels_subset, build=False)
            search_arr.append(search)
            mAP_arr.append(mAP)
            method_arr.append(method)
            recall_arr.append(recall)
            print(f'Run {i} out of {max_percentage} completed for method {method}')

    data_names = ['method', 'searchtime', 'mAP', 'recall']
    data = [method_arr, search_arr, mAP_arr, recall_arr]
    to_csv(data_names, data, filename)


def final_selector_evaluation():
    """
    Evaluates for 100 random queries or for a grid structure the fastest method, mAP and recall
    :return: A print statment telling how many of the random runs are correct
    """
    np.random.seed(1234)
    # Can be adjusted for random/uniform data and the amount of validations
    random = True
    n_validations = 100

    frame_features = np.load((os.path.abspath(r'data/frames.npy')))
    frame_labels = np.load((os.path.abspath(r'data/frames_labels.npy')))
    image_features = np.load((os.path.abspath(r'data/images.npy')))
    image_labels = np.load((os.path.abspath(r'data/images_labels.npy')))
    methods = ['linear', 'faiss_flat_cpu', 'faiss_hnsw', 'faiss_lsh', 'faiss_ivf_cpu']

    if random:
        n_queries = np.random.randint(1, 1000, n_validations)
        n_frames = np.random.randint(1, 50000, n_validations)
    else:
        assert int(np.sqrt(n_validations)) ** 2 == n_validations
        step_size_queries, step_size_frames = int(1000 / int(np.sqrt(n_validations))), int(
            50000 / int(np.sqrt(n_validations)))
        n_queries, n_frames = np.meshgrid(range(1, 1000, step_size_queries), range(1, 50000, step_size_frames))

    fastest_count = n_validations  # decrease every failure
    mAP_recall_count = n_validations  # decrease every failure
    problematic_method_selection = []
    problematic_mAP_recall = []
    for i, (query, frame) in enumerate(zip(n_queries, n_frames)):
        r_queries = np.random.choice(len(image_features), query, False)
        image_features_selec = image_features[r_queries]
        image_labels_selec = image_labels[r_queries]

        r_frames = np.random.choice(len(frame_features), frame, False)
        frame_features_selec = frame_features[r_frames]
        frame_labels_selec = frame_labels[r_frames]

        method_selected = method_selector(frame, query)
        total_time = []

        for method in methods:
            build_time, time_per_query, _, mAP, recall = nn_main_test.nns(frame_features_selec, image_features_selec,
                                                                          method, k_percentage=7,
                                                                          image_labels=image_labels_selec,
                                                                          frame_labels=frame_labels_selec)
            total_time.append(build_time + time_per_query * query)
            if method == method_selected:
                if mAP <= 0.65 or recall <= 0.5:
                    mAP_recall_count -= 1
                    problematic_mAP_recall.append([query, frame, method, mAP, recall])

        method_fastest = methods[total_time.index(min(total_time))]
        if method_fastest != method_selected:
            fastest_count -= 1
            problematic_method_selection.append([query, frame, method_selected, method_fastest])

        print(f"Validation {i + 1}/{n_validations} completed")

    pms = pd.DataFrame(problematic_method_selection,
                       columns=["n_query", "n_frames", "method_selected", "method_fastest"])
    pmr = pd.DataFrame(problematic_mAP_recall, columns=["n_query", "n_frames", "method_selected", "mAP", "recall"])
    pms.to_csv(r".\test_data\problematic_method_selection.csv", index=False)
    pmr.to_csv(r".\test_data\problematic_mAP_recall.csv", index=False)

    print(f"===============================================\n"
          f"Fastest method selection correct for :{fastest_count}/{n_validations}\n"
          f"mAP and recall correct for :{mAP_recall_count}/{n_validations}\n"
          f"===============================================")


if __name__ == "__main__":
    # definition of parser arguments
    a = argparse.ArgumentParser()
    a.add_argument("--n_frames", default=50000, type=int, help="amount of input frames")
    a.add_argument("--n_queries", default=100, type=int, help="amount of queries")
    a.add_argument("--method", default="hnsw_batch", type=str, help="The NNS method to be used")
    a.add_argument("--k", default=100, type=int, help="The amount of nearest neighbors")

    # more for testing purposes
    a.add_argument("--forest_size", default=10, type=int)
    a.add_argument("--annoy_metric", default="angular", type=str)
    a.add_argument("--batch_size", default=500, type=int)
    a.add_argument("--specify_n", default=False, type=bool)
    args = a.parse_args()

    frame_features = np.load((os.path.abspath(r'data/embedded_features.npy')))
    frame_labels = np.load((os.path.abspath(r'data/labels.npy')))
    image_features = np.load((os.path.abspath(r'data/embedded_features_test.npy')))
    image_labels = np.load((os.path.abspath(r'data/labels_test.npy')))

    main()
