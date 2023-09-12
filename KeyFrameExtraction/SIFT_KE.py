import numpy as np
import math
from collections import namedtuple

ColorMoments = namedtuple('ColorMoments', ['mean', 'stdDeviation', 'skewness'])

def getColorMoments(histogram, totalPixels):
    # Computing the color moments
    sum = 0
    for i in range(len(histogram)):
        sum += i*histogram[i]

    # Color moment mean
    mean = float (sum / totalPixels)
    sumOfSquares = 0
    sumOfCubes = 0
    for pixels in histogram:
        sumOfSquares += math.pow(pixels-mean, 2)
        sumOfCubes += math.pow(pixels-mean, 3)

    variance = float (sumOfSquares / totalPixels)

    # Color moment standard deviation
    stdDeviation = math.sqrt(variance)
    avgSumOfCubes = float (sumOfCubes / totalPixels)

    # Color moment skewness
    skewness = float (avgSumOfCubes**(1./3.))
    return ColorMoments(mean, stdDeviation, skewness)


def getEuclideanDistance(currColorMoments, prevColorMoments):

    # Calculate Euclidean Distance of the two color moments
    distance = math.pow(currColorMoments.mean - prevColorMoments.mean, 2) \
               + math.pow(currColorMoments.stdDeviation - prevColorMoments.stdDeviation, 2) \
               + math.pow(currColorMoments.skewness - prevColorMoments.skewness, 2)
    return distance


def colormoments(descriptor, shot_frame_number, totalpixels):

    # Declaring value
    euclideanDistance = []

    # Get initial color moment
    prevColorMoments = getColorMoments(descriptor[0], totalpixels)

    # Computing Euclidean distances of every frame
    for i in range(1, len(descriptor)):

        colorMoments = getColorMoments(descriptor[i], totalpixels)
        euclideanDistance.append( getEuclideanDistance(colorMoments, prevColorMoments) )
        prevColorMoments = colorMoments

    # Maximum keyframes taken
    perc = 2
    # Multiplication factor of threshold
    factor = 6
    keyFramesIndices = []

    meanEuclideanDistance = sum(euclideanDistance[1:]) / float(len(euclideanDistance) - 1)
    thresholdEuclideanDistance = factor * meanEuclideanDistance

    for i in range(len(euclideanDistance)):
        if euclideanDistance[i] >= thresholdEuclideanDistance:
            keyFramesIndices.append(i)

    # If the number of keyframes taken using the threshold is greater than
    # the maximum percentage keyframes taken, only the highest percentage frames will be taken
    if len(keyFramesIndices) > i*perc/100:
        keyFramesIndices = sorted(np.argsort(euclideanDistance)[::-1][:int(max(i*perc/100,1))])

    keyframe_indices = np.array(keyFramesIndices)

    # Normalizing the keyframe indices
    keyframe_indices = [(element + shot_frame_number) for element in keyframe_indices]

    return keyframe_indices