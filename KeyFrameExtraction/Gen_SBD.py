import copy
import time
from basicmethods import *
from histogramblockclustering import *
from VSUMM_KE import *
from VSUMM_combi import *
from descriptors import *
from SIFT_KE import *

__hist_size__ = 128             # how many bins for each R,G,B histogram
__min_duration__ = 10           # if a shot has length less than this, merge it with others
__CFAR__ = False
__SBD_method__ = "HBT"
__PBT_method__ = "all"

def SBD(cap, method, performSBD, presample, video_fps):
    """
    Performs shot based detection and calls keyframe extraction method to return indices
    :param cap: the capture of input video
    :param method: the method of keyframe extraction after performing shot detection (crudehistogram, firstmiddlelast, firstlast, firstonly, histogramblockclustering, VSUMM, VSUMM_combi, colormoments)
    :param performSBD: boolean, perform shot boundary detection or not
    :param presample: boolean, presample the input video for speed gain (default  = 10 fps)
    :param video_fps: the framerate of input video
    :return: indices of keyframes
    """

    # set timer
    time_SBD = time.time()


    sampling_rate = 10 # presampling to decrease computation time for reading frames
    skip_num = video_fps/sampling_rate # retrieve every n-th frame

    if not performSBD:
        print("No SBD performed! Viewing video as one entire shot/segment")
        frame_count = 0

        # Empty array to put feature descriptors in for chosen method
        method_descriptors = []

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        totalpixels = width*height

        if presample:   #grab every n-thm frame
            count = 0
            while True:
                success = cap.grab()
                if not success:
                    break
                if int(count % skip_num) == 0:
                    ret, frame = cap.retrieve()
                    method_descriptors.append(createDescriptor(method, frame))  #add descriptor for current frame
                    frame_count += 1
                count += 1
        else:
            while True:
                success, frame = cap.read()
                if not success:
                    break
                method_descriptors.append(createDescriptor(method, frame))  #add descriptor for current frame
                frame_count += 1


        shot_boundary = [0, frame_count]
        print("Shot boundaries: " + str(shot_boundary))
        print('\033[94m' + f'Time to read (presampled) video and  generate descriptors for chosen method: {time.time() - time_SBD}' + '\033[0m')

    else: #  perform SBD

        if __SBD_method__ == "PBT":
            totalpixels, method_descriptors, shot_boundary = PBT(method, cap, presample, skip_num)
        else:
            totalpixels, method_descriptors, shot_boundary = HBT(method, cap, presample, skip_num)

        if presample:
            actual_shot_boundary = [round(element * skip_num) for element in shot_boundary]
            print(f'>>> There are {len(actual_shot_boundary) - 1} shots found at  {actual_shot_boundary}')
        else:
            print(f'>>> There are {len(shot_boundary) - 1} shots found at  {shot_boundary}')
        print('\033[94m' + f'Time to apply SBD and generate descriptors for chosen method: {time.time() - time_SBD}'+ '\033[0m')
        print(">>> Applying " + method + " to shots")

    keyframe_indices = []
    for i in range(0, len(shot_boundary) - 1):
        keyframe_indices.append(KFE(presample, skip_num, method, method_descriptors[int(shot_boundary[i]):int(shot_boundary[i + 1] - 1)], int(shot_boundary[i]), totalpixels))
    return keyframe_indices




def KFE(presample, skip_num, method, method_descriptors, shot_frame_number,totalpixels):
    """
    Applies chosen method to frames in shot using descriptors that were generated
    :return: indices of keyframes
    """
    if method == "crudehistogram":
        keyframe_indices = histogram_summary(method_descriptors, shot_frame_number)
    elif method == "firstmiddlelast":
        keyframe_indices = first_middle_last(method_descriptors, shot_frame_number)
    elif method == "firstlast":
        keyframe_indices = first_last(method_descriptors, shot_frame_number)
    elif method == "firstonly":
        keyframe_indices = first_only(method_descriptors, shot_frame_number)
    elif method == "histogramblockclustering":
        keyframe_indices = blockclustering(method_descriptors, shot_frame_number)
    elif method == "shotdependentsampling":
        keyframe_indices = shotdependent_sampling(method_descriptors, shot_frame_number)
    elif method == "VSUMM":
        keyframe_indices = VSUMM(method_descriptors, shot_frame_number)
    elif method == "VSUMM_combi":
        keyframe_indices = VSUMM_combi(method_descriptors, shot_frame_number, skip_num)
    elif method == "colormoments":
        keyframe_indices = colormoments(method_descriptors, shot_frame_number, totalpixels)

    # multiply every index with skip_num if pre-sampling was performed to get correct indices
    if presample:
        keyframe_indices = [round(element * skip_num) for element in keyframe_indices]

    return keyframe_indices



def PBT(method, cap, presample, skip_num):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    quotient = 256 * width * height
    totalpixels = width * height

    shot_boundary = []
    shot_boundary.append(0)
    framediff = []
    method_descriptors = []
    success, old_frame = cap.read()
    method_descriptors.append(createDescriptor(method, old_frame))
    old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    n_frames = 0

    if presample:
        min_dur = int(__min_duration__/skip_num)
        count = 0
        while True:
            success = cap.grab()
            if not success:
                break
            if int(count % skip_num) == 0:
                ret, frame = cap.retrieve()
                method_descriptors.append(createDescriptor(method, frame))
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_old = np.array(old_frame) / quotient
                frame_new = np.array(frame_gray) / quotient

                if __PBT_method__ == "all":
                    framediff.append(np.abs(np.sum(frame_new) - np.sum(frame_old)))
                else:
                    framediff.append(np.sum(np.abs(np.subtract(frame_new, frame_old))))
                old_frame = frame_gray
                n_frames += 1
            count += 1
    else:
        min_dur = int(__min_duration__)
        success, frame = cap.read()
        while success:
            n_frames += 1
            method_descriptors.append(createDescriptor(method, frame))

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_old = np.array(old_frame)/quotient
            frame_new = np.array(frame_gray)/quotient

            if __PBT_method__ == "all":
                framediff.append(np.abs(np.sum(frame_new) - np.sum(frame_old)))
            else:
                framediff.append(np.sum(np.abs(np.subtract(frame_new, frame_old))))
            old_frame = frame_gray
            success, frame = cap.read()


    mean_diff = sum(framediff) / len(framediff)
    frame_index = []

    if __CFAR__ == False:
        factor = 6
        for idx in range(len(framediff)):
            if framediff[idx] > factor * mean_diff:
                frame_index.append(idx)
    else:
        if __PBT_method__ == "all":
            factor = 6
        else:
            factor = 2
        conv = np.ones(5)
        diffconv = np.convolve(framediff, conv) / 2

        for j in range(len(framediff)):
            if factor * mean_diff > diffconv[j]:
                convthresh = factor * mean_diff
            else:
                convthresh = diffconv[j]

            if framediff[j] >= convthresh:
                frame_index.append(j)


    tmp_idx = copy.copy(frame_index)
    for i in range(0, len(frame_index) - 1):
        if frame_index[i + 1] - frame_index[i] < min_dur:
            del tmp_idx[tmp_idx.index(frame_index[i])]
    frame_index = tmp_idx

    # the real index start from 1 but time 0 and end add to it
    idx_new = copy.copy(frame_index)
    idx_new.insert(0, -1)
    # if n_frames - 1 - idx_new[-1] < min_dur:
    #     del idx_new[-1]

    idx_new.append(n_frames - 1)

    idx_new = list(map(lambda x : x + 1, idx_new))

    return totalpixels, method_descriptors, idx_new


def HBT(method, cap, presample, skip_num):

    hists = []
    method_descriptors = []
    n_frames = 0

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    totalpixels = width*height
    if presample:
        min_dur = int(__min_duration__/skip_num)
        count = 0
        while True:
            success = cap.grab()
            if not success:
                break
            if int(count % skip_num) == 0:
                ret, frame = cap.retrieve()
                chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0, 256]) \
                          for c in range(3)]
                chists = np.array([chist for chist in chists])
                hists.append(chists.flatten())
                method_descriptors.append(createDescriptor(method, frame))
                n_frames += 1
            count += 1
    else:
        min_dur = int(__min_duration__)
        success, frame = cap.read()
        while success:
            chists = [cv2.calcHist([frame], [c], None, [__hist_size__], [0,256]) \
                          for c in range(3)]
            chists = np.array([chist for chist in chists])
            hists.append(chists.flatten())
            method_descriptors.append(createDescriptor(method, frame))
            n_frames += 1
            success, frame = cap.read()

    # compute hist  distances
    scores = [np.ndarray.sum(abs(pair[0] - pair[1])) for pair in zip(hists[1:], hists[:-1])]

    average_frame_div = sum(scores) / len(scores)
    frame_index = []

    if __CFAR__ == False:
        factor = 6
        for idx in range(len(scores)):
            if scores[idx] > factor * average_frame_div:
                frame_index.append(idx)
    else:
        factor = 3
        conv = np.ones(5)
        diffconv = np.convolve(scores, conv) / 2

        for j in range(len(scores)):
            if factor * average_frame_div > diffconv[j]:
                convthresh = factor * average_frame_div
            else:
                convthresh = diffconv[j]

            if scores[j] > convthresh:
                frame_index.append(j)


    tmp_idx = copy.copy(frame_index)
    for i in range(0, len(frame_index) - 1):
        if frame_index[i + 1] - frame_index[i] < min_dur:
            del tmp_idx[tmp_idx.index(frame_index[i])]
    frame_index = tmp_idx

    # the real index start from 1 but time 0 and end add to it
    idx_new = copy.copy(frame_index)
    idx_new.insert(0, -1)
    # if n_frames - 1 - idx_new[-1] < min_dur:
    #     del idx_new[-1]
    #print(self.n_frames)
    idx_new.append(n_frames - 1)

    idx_new = list(map(lambda x : x + 1, idx_new))

    return totalpixels, method_descriptors, idx_new




