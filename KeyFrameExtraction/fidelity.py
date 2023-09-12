from main import *
from scipy.spatial import distance
from math import atan2

def fidelity_descriptors(path):
    """
    Generates histogram descriptors to compute fidelity with
    :param path: video path
    :return: fd_data, hist_data, fdnorm, histnorm (histogram descriptors along with norms to normalize)
    """
    cap = cv2.VideoCapture(path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    histnorm = width * height  # to normalize color histogram with
    downscale = 0.5  # downsize frame for lower computation for hogs
    fdnorm = histnorm * (downscale) ** 2  # to normalize edge detection histogram

    fd_data = []
    hist_data = []
    cnt = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        # print("Creating histograms for frame " + str(cnt))
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        hist_data.append(hist)
        fd_data.append(fd)
        cnt += 1
        if cnt % 50 == 0:
            print("created hog for frame: " + str(cnt))
    print("end of histogram creation")
    return fd_data, hist_data, fdnorm, histnorm

def fidelity(kf_indices, path, vseq_hists, vseq_hogs, fdnorm, histnorm):
    """
    Computes fidelity for a chosen selection of keyframes
    :param kf_indices: indices of keyframes
    :param path: path of video (won't be used here anymore)
    :param vseq_hists: (color) histograms of all frames in video
    :param vseq_hogs: (edge) histograms of all frames in video
    :param fdnorm: norm of edge direction histogram
    :param histnorm: norm of color histogram (amount of pixels)
    :return: Fidelity value (within [0, 1])
    """

    # Semi Hausdorff distance
    maxdiff = 1 # maximum value difference() can return
    maxdist = 0
    for i in range(0, len(vseq_hists)):
        distance = maxdiff
        for j in range(0, len(kf_indices)):
            diff = difference(vseq_hists[i], vseq_hists[kf_indices[j]], vseq_hogs[i], vseq_hogs[kf_indices[j]], fdnorm, histnorm)
            if diff < distance:
                distance = diff
        if (distance > maxdist):
            maxdist = distance

    return maxdiff - maxdist

def calculateHOG(frame, downsize):
    """
    OBSOLETE: descriptors are generated directly
    :param frame:
    :param downsize:
    :return:
    """
    #resize image for computational speed gain
    scale_percent = downsize  # percent of original size
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    # Gaussian blur (kernel size = 3)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    # Convert to grayscale (= luminance)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    gradientmax = 255 * 4  # maximum gradient value
    threshold = gradientmax * 0.3  # threshold to reduce background noise
    # apply sobel filters,
    grad_x = np.array(cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))
    grad_y = np.array(cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT))

    # calculate absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gradient magnitude calculated by 0.5(|G_x|+|G_y|) instead of sqrt(G_x^2 + G_y^2)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    histogram = np.zeros(72, dtype="float32")
    for i in range(0, grad.shape[0]):
        for j in range(0, grad.shape[1]):
            if grad[i][j]:
                angle = atan2(grad_y[i][j], grad_x[i][j]) / 3.14
                histogram[round(abs(angle * 71))] += int(1)
    return histogram

def calculateHists(keyframes, path, video_fps, cap, downscale):
    """
    OBSOLETE
    :param keyframes:
    :param path:
    :param video_fps:
    :param cap:
    :param downscale:
    :return:
    """
    #parameters

    fd_sel = []
    hist_sel = []
    print("Creating hogs")
    for frame in keyframes:
        #print("Creating hog for frame")
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        hist_sel.append(hist)
        fd_sel.append(fd)

    fd_data = []
    hist_data = []
    cnt = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        #print("Creating histograms for frame " + str(cnt))
        fd = calculateHOG(frame, downscale)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
        hist_data.append(hist)
        fd_data.append(fd)
        cnt += 1
        if cnt%50 == 0:
            print("created hog for frame: " + str(cnt))
        # if cnt > 10:
        #     break
    print("end of histogram creation")
    return fd_sel, fd_data, hist_sel, hist_data


def difference(hist1, hist2, fd1, fd2, fd_norm, hist_norm):
    """
    Computes distance between 2 frames
    :param hist1:
    :param hist2:
    :param fd1:
    :param fd2:
    :param fd_norm:
    :param hist_norm:
    :return:
    """
    d_h = 1-cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)/hist_norm
    #print(d_h)
    #d_d = 1-cv2.compareHist(fd1, fd2, cv2.HISTCMP_INTERSECT)/fd_norm

    # normalize feature descriptors edges
    n_fd = fd1 / np.sqrt(np.sum(fd1 ** 2))
    n_fd2 = fd2 / np.sqrt(np.sum(fd2 ** 2))

    d_d = distance.euclidean(n_fd, n_fd2)


    return d_h*d_d #originally d_h*d_w + d_h*d_d + d_d*d_w