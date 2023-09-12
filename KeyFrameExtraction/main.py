import sys
import os
import time

from Gen_SBD import *
from fidelity import *

def save_keyframes(keyframe_indices, frames_data, savepath = "keyframes"):
    """
    saves (key)frames into folder
    :param keyframe_indices: indices of frame which will become file names
    :param frames_data: the corresponding RGB-data
    :param savepath: path to save folder, default is /keyframes
    """
    print("Extracting keyframes to " + str(savepath))
    #savepath = os.path.expanduser("~/bin/keyframes")

    try:
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    except OSError:
        print("Error can't make directory")
    for i in range (0, len(keyframe_indices)):
        # frame_rgb = cv2.cvtColor(frames_data[i], cv2.COLOR_BGR2RGB)
        frame_rgb = frames_data[i]
        #print("Extracting frame " + str(keyframe_indices[i]))
        name = savepath + '/' + str(keyframe_indices[i]) + '.jpg'
        cv2.imwrite(name, frame_rgb)

def keyframe_extraction(video_path, method, performSBD, presample):
    """
    Performs the extraction of keyframes of an input video
    :param video_path: the path to the input video
    :param method: the method of keyframe extraction after performing shot detection (crudehistogram, firstmiddlelast, firstlast, firstonly, histogramblockclustering, VSUMM, VSUMM_combi, colormoments)
    :param performSBD: boolean, perform shot boundary detection or not
    :param presample: boolean, presample the input video for speed gain (default  = 10 fps)
    :return: Indices of keyframes, corresponding rgb-data and video_fps
    """

    print("Opening video: " + video_path)
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception("Error opening video stream")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    method_time = time.time()

    print('>>> Performing Shot Based Detection')
    keyframes_indices = SBD(cap, method, performSBD, presample, video_fps)

    # Convert [[x,x,x], [x,x,x,x], [x,x,x,x], ... ] to one axis array
    keyframes_idx = []
    for i in range(0, len(keyframes_indices)):
        for j in range(0, len(keyframes_indices[i])):
            keyframes_idx.append(keyframes_indices[i][j])

    print('\033[93m' + f'Time to select indices with method using descriptors: {time.time() - method_time}'+ '\033[0m')
    keyframes_data = keyframes_indices_to_array(keyframes_idx, video_path, frame_count, video_fps)

    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps



def KE_uniform_sampling(video_path, rate, CR):
    """
    Performs the extraction of keyframes of an input video by means of uniform sampling
    Similar to keyframe_extraction() but skips SBD and presampling and generates indices directly
    :param video_path: the path to the input video
    :param rate: sampling rate
    :param CR: compression ratio
    :return Indices of keyframes, corresponding rgb-data and video_fps
    """
    # Generates indices and gets the framedata from the video using CAP_PROP_POS
    # CAP_PROP_POS is only superior to regular reading ad discarding for uniformly sampling if every 21-th frame or more
    # (=1.4fps for a 30fps video) is taken

    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception("Error opening video stream")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    method_time = time.time()

    choose_rate = True   # set to False for Compression Ratio instead

    if choose_rate:
        skip_num = video_fps / rate # every k-th frame should be taken
    else:
        skip_num = 1/(1-CR)

    # generate indices
    keyframes_idx = [i for i in range(frame_count-1) if int(i % skip_num) == 0]

    print('\033[93m' + f'Time to select indices with uniform sampling: {time.time()-method_time}'+ '\033[0m')

    keyframes_data = keyframes_indices_to_array(keyframes_idx, video_path, frame_count, video_fps)

    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps


def fast_uniform_sampling(video_path, rate, CR):
    """
    obsolete script; general KE_uniform_sampling is faster for all rates
    Reads video and stores frames uniformly and discards others
    Reading and discarding is better for faster sampling than setting the frame location to be picked using CAP_PROP_POS
    """
    print('>>> Uniformly sampling frames from video directly into array w/o selected keyframe indices')
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        raise Exception("Error opening video stream")
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    choose_rate = True  # set to False for Compression Ratio instead
    if choose_rate:
        skip_num = video_fps / rate
    else:
        skip_num = 1/(1-CR)

    keyframes_data = []
    keyframes_idx = []

    count = 1
    while True:
        success = cap.grab()
        if not success:
            break
        if int(count%skip_num) == 0:
            ret, frame = cap.retrieve()
            keyframes_data.append(frame)
            keyframes_idx.append(count)
        count += 1
    print_statistics(frame_count, video_fps, keyframes_idx)

    return keyframes_data, keyframes_idx, video_fps


def keyframes_indices_to_array(indices, path, frame_count, video_fps):
    """
    Decodes selected frames indices to array
    :param indices: indices of frames to be decoded
    :param path: the path to the input video
    :param frame_count: the amount of frames in the input video
    :param video_fps: fps of original video
    :return: RGB-data of frames
    """
    print('>>> Decoding selected frames from video into array')
    frames = []
    cap = cv2.VideoCapture(path)
    if (cap.isOpened() == False):
        raise Exception("Error opening video stream")

    summary_ratio = 0.70/video_fps # about 0.22 of the amount of frames as threshold
    # for CR lower than 97.8%, CAP_PROP_POS_FRAMES is slower than reading and discarding undesired frames

    if (len(indices) < frame_count*summary_ratio):  #use CAP_PROP_POS_FRAMES
        for i in range(0, len(indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i])
            _, frame = cap.read()
            frames.append(frame)
    else:                                           #use grab() and retrieve()
        count = 0
        idx = 0
        while True:
            success = cap.grab()
            if not success:
                break
            if (count == indices[idx]):
                ret, frame = cap.retrieve()
                frames.append(frame)
                idx += 1
            if idx > len(indices)-1:
                break
            count += 1
    return frames

def print_statistics(frame_count, video_fps, keyframes_idx):
    """
    Prints statistics about the keyframe set compared to input video
    :param frame_count: frame count of input video
    :param video_fps: framerate of input video
    :param keyframes_idx: indices of keyframes (array)
    """

    print(f'>>> There are {len(keyframes_idx)} keyframes extracted (indices: {keyframes_idx}).')
    print("STATISTICS:")
    duration = frame_count/video_fps
    compression = 1- len(keyframes_idx)/frame_count
    frames_per_second = len(keyframes_idx)/duration
    print("Duration of input video: " + str(duration))
    print("Avg. keyframes per second: " + str(frames_per_second))
    print("Compression Ratio: " + str(compression))


if __name__ == '__main__':
    KE_method = "VSUMM_combi"
    performSBD = True
    presample = False
    kfe_time = time.time()
    keyframes_data, keyframe_indices, video_fps = keyframe_extraction(sys.argv[1], KE_method, performSBD, presample)
    #keyframes_data, keyframe_indices, video_fps = KE_uniform_sampling(sys.argv[1], 9, 0.85)
    print('\033[92m' + f' Total KeyframeExtraction time: {time.time() - kfe_time}' + '\033[0m')
    save_keyframes(keyframe_indices, keyframes_data)