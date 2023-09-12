import optuna
import os
import sys
import numpy as np
import argparse
import joblib

from nn_faiss import faiss_lsh
from nn_faiss import faiss_ivf
from nn_faiss import faiss_hnsw

from nn_main_test import cal_mAP
from nn_main_test import cal_recall

frame_features = np.load((os.path.abspath(r'data/frames.npy')))
frame_labels = np.load((os.path.abspath(r'data/frames_labels.npy')))
image_features = np.load((os.path.abspath(r'data/images.npy')))
image_labels = np.load((os.path.abspath(r'data/images_labels.npy')))

def optimizelsh(trial):
    '''
    Defines the objective function for the HPO.
    :param trial:
    :return: mAP, recall, time per query, build time
    '''
    # HPO for FAISS LSH: Define range of values for bitlength_percentage and call nns method
    bitlength_percentage = trial.suggest_discrete_uniform("bitlength_percentage",0,1,0.01)
    idx, dist, build_time, time_per_query = faiss_lsh(image_features, frame_features, k, bitlength_percentage)

    # Calculate map and recall
    mAP = cal_mAP(idx,frame_labels,image_labels)
    recall = cal_recall(idx, frame_labels,image_labels)
    return mAP, recall, time_per_query, build_time

def optimizeivf(trial):
    '''
    Defines the objective function for the HPO.
    :param trial:
    :return: mAP, recall, time per query, build time
    '''

    # HPO for FAISS IVF: Define range of values for splits and nprobe and call nns method
    max_splits = int(np.log2(args.n_frames))
    splits = trial.suggest_int("splits",1,max_splits,1)
    nprobe = trial.suggest_int("nprobe",1,splits,1)
    idx, dist, build_time, time_per_query = faiss_ivf(image_features, frame_features, k, False, splits, nprobe)

    # Calculate map and recall
    mAP = cal_mAP(idx,frame_labels,image_labels)
    recall = cal_recall(idx, frame_labels,image_labels)
    return mAP, recall, time_per_query, build_time

def optimizehnsw(trial):
    '''
    Defines the objective function for the HPO.
    :param trial:
    :return: mAP, recall, time per query, build time
    '''

    # HPO for FAISS HNSW: Define range of values for m and ef_const and call nns method
    m = trial.suggest_int("m",2,32,1)
    ef_const = trial.suggest_int("ef_const",1,10000,1)
    print(m,ef_const)
    idx, dist, build_time, time_per_query = faiss_hnsw(image_features, frame_features, k, m, ef_const)

    # Calculate map and recall
    mAP = cal_mAP(idx,frame_labels,image_labels)
    recall = cal_recall(idx, frame_labels,image_labels)
    return mAP, recall, time_per_query, build_time


def main():
    '''
    Performs the HPO study
    :return:
    '''
    # create study where we try to maximize mAP, maximize recall, minimize time_per_query, minimize build_time
    study_name = f"{args.method}"
    study = optuna.create_study(directions=["maximize", "maximize", "minimize", "minimize"])
    # perform HPO for selected objective function
    study.optimize(optimizelsh, n_trials=25)
    # save all results to pickle files
    joblib.dump(study, f"./test_data/hpo_results/{args.method}{args.n_frames}.pkl")

if __name__ == "__main__":
    # definition of parser arguments
    a = argparse.ArgumentParser()
    a.add_argument("--n_frames", default=50000, type=int, help="amount of input frames")
    a.add_argument("--method", type=str, help="method")
    args = a.parse_args()

    np.random.seed(1234)  # for generating consistent data
    # params to be set dependent on dataset (this case CIFAR-10)
    if args.n_frames < 50000:
        dataset_new_size = args.n_frames
        dataset_n_classes = 10
        new_dataset_indices = []
        assert dataset_new_size % dataset_n_classes == 0

        for i in range(dataset_n_classes):
            new_dataset_indices.append(
                np.random.choice(np.where(frame_labels == i)[0], int(dataset_new_size / dataset_n_classes),
                                 replace=False))

        new_dataset_indices_flatten = np.array(new_dataset_indices).flatten()
        frame_features = frame_features[new_dataset_indices_flatten]
        frame_labels = frame_labels[new_dataset_indices_flatten]
    k = round(0.07 * args.n_frames)
    main()
