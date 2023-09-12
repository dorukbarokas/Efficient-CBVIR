import numpy as np
import cv2


def histogram_summary(hists, shot_start_number):
    """
    Generates array of keyframe indices using descriptors of histgrams using fixed threshold
    :param hists: histogram vectors of frames
    :param shot_start_number: start index of shot
    :return: indices
    """

    # intersection threshold
    threshold = 0.3

    current_keyframe_hist = hists[0]
    keyframe_indices = [shot_start_number]
    for i in range(1, len(hists)):
        histdiff = cv2.compareHist(current_keyframe_hist, hists[i], cv2.HISTCMP_BHATTACHARYYA)

        if (histdiff) > threshold:

            keyframe_indices.append(i+shot_start_number)
            current_keyframe_hist = hists[i]

    return keyframe_indices


def first_middle_last(descriptors, shot_start_number):
    # Generates array of indices using the shot index and number of frames in shot
    print('Selecting first, middle and last frame from shot')
    middle = int((shot_start_number+len(descriptors)/2))
    indices = [shot_start_number, middle, shot_start_number+len(descriptors)]
    return indices

def first_last(descriptors, shot_start_number):
    # Generates array of indices using the shot index and number of frames in shot
    print('Selecting first and last frame from shot')
    indices = [shot_start_number, shot_start_number+len(descriptors)]
    return indices

def first_only(descriptors, shot_start_number):
    # Generates array of indices using the shot index
    print('Selecting first frame from shot')
    indices = [shot_start_number]
    return indices

def shotdependent_sampling(descriptors, shot_start_number):
    indices = [shot_start_number]
    return indices

def changeIdxFormat(indices, frame_count):
    # Converts the format of keyframe selection from [0, 3, 5 ... ] to [0., 0., 0., 0., 1, 0., 1, ... ]
    print("Converting KF selection [a, b, c, ..., d] to [0, 0, 1, 0 ... 1 ] ")
    sumsel = np.zeros([frame_count, 1])
    for keyframe in indices:
        sumsel[keyframe] = 1
    return sumsel

