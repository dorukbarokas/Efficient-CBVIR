import cv2
import numpy as np

def createDescriptor(method, frame):
    """
    Generates single descriptor for a frame using specified method
    :param method: Chosen keyframe extraction method
    :param frame: BGR-data of a frame
    :return: descriptor number/vector
    """

    if method == "crudehistogram":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # calculate 8 bin histogram
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        # flatten histogram
        hist = cv2.normalize(hist, None).flatten()
        descriptor = hist
    if method == "histogramblockclustering":
        frame_rgb = frame

        # dividing a frame into 3*3 i.e 9 blocks
        height, width, channels = frame_rgb.shape

        if height % 3 == 0:
            h_chunk = int(height / 3)
        else:
            h_chunk = int(height / 3) + 1
        if width % 3 == 0:
            w_chunk = int(width / 3)
        else:
            w_chunk = int(width / 3) + 1
        h = 0
        w = 0
        feature_vector = []
        for a in range(1, 4):
            h_window = h_chunk * a
            for b in range(1, 4):
                frame = frame_rgb[h: h_window, w: w_chunk * b, :]
                hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6],
                                    [0, 256, 0, 256, 0, 256])  # finding histograms for each block
                hist1 = hist.flatten()  # flatten the hist to one-dimensinal vector
                feature_vector += list(hist1)
                w = w_chunk * b

            h = h_chunk * a
            w = 0


        descriptor = feature_vector # 1944 one dimensional feature vector for frame

    elif method == "VSUMM" or method == "VSUMM_combi":
        channels=['b','g','r']
        num_bins = 16
        feature_value=[cv2.calcHist([frame],[i],None,[num_bins],[0,256]) for i,col in enumerate(channels)]
        descriptor = np.asarray(feature_value).flatten()

    elif method == "firstmiddlelast" or method == "firstonly" or method == "firstlast" or method == "uniformsampling" or method == "shotdependentsampling":
        descriptor = 1

    elif method == "colormoments":
        grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        descriptor= cv2.calcHist([grayImage], [0], None, [256], [0, 256])

    return descriptor