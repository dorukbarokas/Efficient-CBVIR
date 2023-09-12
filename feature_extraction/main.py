from PIL import Image
import numpy as np
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.abspath('solar/solar_global')))
from fe_main import extract_features_global


def main():
    test_inputs_outputs()


def test_inputs_outputs():
    # Read in test images.
    frame_paths = glob.glob(r"D:\Thesis\historical\keyframes\video_*.jpg")

    # frame_paths = (f'/home/aron/Downloads/fe/test-set/ewi-tudelft-logo/frames/frame_{x:02d}.jpg'
    #                for x in range(0, 55))
    image_paths = ['D:/Thesis/historical/keyframes/video_0_2_19.jpg']
    search_images = []
    frame_images = []
    for path in image_paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.array(img)
        if img.shape[2] == 4:
            img = img[..., :3]
        search_images.append(img)
    for path in frame_paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = np.array(img)
        if img.shape[2] == 4:
            img = img[..., :3]
        frame_images.append(img)

    search_images = np.array(search_images)
    frame_images = np.array(frame_images)

    # feed the images to the feature extraction
    size = 576
    search_features = extract_features_global(search_images, size)
    frame_features = extract_features_global(frame_images, size)

    dist = frame_features - search_features[:1]  # for now only consider the first search image
    dist = np.linalg.norm(dist, axis=1)
    res = np.argsort(dist)

    dist = dist[res]

    print(f'Res has length {len(res)}.\n{res}')
    print(f'Dist:\n{dist}')


if __name__ == '__main__':
    main()
