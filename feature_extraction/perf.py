from PIL import Image
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('solar/solar_global')))
from fe_main import extract_features_global


def main():
    names = [
        'Battuta1',
        'Battuta1',
        'Battuta1',
        'Battuta1',

        'Battuta2',

        'He1',
        'He1',
        'He1',

        'He2',

        'He3',
        'He3',
        'He3',

        'Polo1',

        'tudelft-ewi-1',
        'tudelft-ewi-2',
        'dutch-mailbox-2',

        'smirnoff-ice-crop',
        'smirnoff-ice-crop-logo'
    ]
    lst_frame_paths = [
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/frame_{x:03d}.jpg'
         for x in range(0, 524)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/frame_{x:03d}.jpg'
             for x in range(0, 524)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/frame_{x:03d}.jpg'
             for x in range(0, 524)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/frame_{x:03d}.jpg'
             for x in range(0, 524)),

        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Battuta2/frame_{x:03d}.jpg'
             for x in range(0, 379)),

        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He1/frame_{x:03d}.jpg'
             for x in range(0, 550)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He1/frame_{x:03d}.jpg'
             for x in range(0, 550)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He1/frame_{x:03d}.jpg'
             for x in range(0, 550)),

        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He2/frame_{x:03d}.jpg'
             for x in range(0, 228)),

        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He3/frame_{x:03d}.jpg'
             for x in range(0, 377)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He3/frame_{x:03d}.jpg'
             for x in range(0, 377)),
        list(f'/home/aron/Videos/traveller-videos-sparse/videos/He3/frame_{x:03d}.jpg'
             for x in range(0, 377)),

        list(f'/home/aron/Videos/traveller-videos-sparse/videos/Polo1/frame_{x:03d}.jpg'
             for x in range(0, 649)),

        list(f'/home/aron/Videos/deja-vu-dataset/videos/tudelft-ewi-1/frame_{x:03d}.jpg'
             for x in range(0, 89)),
        list(f'/home/aron/Videos/deja-vu-dataset/videos/tudelft-ewi-2/frame_{x:03d}.jpg'
             for x in range(0, 77)),
        list(f'/home/aron/Videos/deja-vu-dataset/videos/dutch-mailbox-2/frame_{x:03d}.jpg'
             for x in range(0, 112)),

        list(f'/home/aron/Videos/deja-vu-dataset/videos/ice-inside-2-shot/frame_{x:03d}.jpg'
             for x in range(0, 108)),
        list(f'/home/aron/Videos/deja-vu-dataset/videos/ice-inside-2-shot/frame_{x:03d}.jpg'
             for x in range(0, 108))
    ]
    lst_image_path = [
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/1.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/2.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/3.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/4.png',

        '/home/aron/Videos/traveller-videos-sparse/images/all_images/81.png',

        '/home/aron/Videos/traveller-videos-sparse/images/all_images/22.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/23.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/24.png',

        '/home/aron/Videos/traveller-videos-sparse/images/all_images/405.png',

        '/home/aron/Videos/traveller-videos-sparse/images/all_images/411.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/412.png',
        '/home/aron/Videos/traveller-videos-sparse/images/all_images/413.png',

        '/home/aron/Videos/traveller-videos-sparse/images/all_images/11.png',

        '/home/aron/Videos/deja-vu-dataset/images/ewi-tudelft-1.jpg',
        '/home/aron/Videos/deja-vu-dataset/images/ewi-tudelft-1.jpg',
        '/home/aron/Videos/deja-vu-dataset/images/dutch-mailbox.jpg',

        '/home/aron/Videos/deja-vu-dataset/images/empty-ice-crop.jpg',
        '/home/aron/Videos/deja-vu-dataset/images/empty-ice-crop-logo.jpg'
    ]
    lst_labels_file = [
        '/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/labels-ranges-1.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/labels-ranges-2.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/labels-ranges-3.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/Battuta1/labels-ranges-4.csv',

        '/home/aron/Videos/traveller-videos-sparse/videos/Battuta2/labels-ranges-81.csv',

        '/home/aron/Videos/traveller-videos-sparse/videos/He1/labels-ranges-22.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/He1/labels-ranges-23.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/He1/labels-ranges-24.csv',

        '/home/aron/Videos/traveller-videos-sparse/videos/He2/labels-ranges-405.csv',

        '/home/aron/Videos/traveller-videos-sparse/videos/He3/labels-ranges-411.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/He3/labels-ranges-412.csv',
        '/home/aron/Videos/traveller-videos-sparse/videos/He3/labels-ranges-413.csv',

        '/home/aron/Videos/traveller-videos-sparse/videos/Polo1/labels-ranges-11.csv',

        '/home/aron/Videos/deja-vu-dataset/videos/tudelft-ewi-1/labels.csv',
        '/home/aron/Videos/deja-vu-dataset/videos/tudelft-ewi-2/labels.csv',
        '/home/aron/Videos/deja-vu-dataset/videos/dutch-mailbox-2/labels.csv',

        '/home/aron/Videos/deja-vu-dataset/videos/ice-inside-2-shot/labels.csv',
        '/home/aron/Videos/deja-vu-dataset/videos/ice-inside-2-shot/labels.csv'
    ]
    lst_out_file = [
        '/home/aron/repos/ibvse/featureextraction/measurements/battuta1_1.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/battuta1_2.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/battuta1_3.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/battuta1_4.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/battuta2_88.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/he1_22.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/he1_23.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/he1_24.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/he2_405.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/he3_411.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/he3_412.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/he3_413.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/polo1_11.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/tudelft-ewi-1.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/tudelft-ewi-2.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/dutch-mailbox-2.csv',

        '/home/aron/repos/ibvse/featureextraction/measurements/empty-ice-crop.csv',
        '/home/aron/repos/ibvse/featureextraction/measurements/empty-ice-crop-logo.csv'
    ]

    assert len(names) == len(lst_frame_paths) == len(lst_image_path) == len(lst_labels_file) == \
        len(lst_out_file), 'the lists should be of the same length'

    for n, f_p, i_p, l_f, o_f in zip(names, lst_frame_paths, lst_image_path, lst_labels_file, lst_out_file):
        test_inputs_outputs(n, f_p, i_p, l_f, o_f)


def test_inputs_outputs(name, frame_paths, image_path, labels_file, out_file):
    """Get the oh so important performance stats

    Arguments:
        frame_paths:    generator for the frame paths
        image_path:     path to the input image
        labels_file:    path to the labels file corresponding to the input image
        out_file:       path to use for the output file
    """
    data = {
        'name': [],
        'search_image': [],
        'num_of_frames': [],
        'ap': [],
        'recall': [],
        'k': [],
        'size': []
    }
    sizes = [144, 180, 240, 360, 480, 540, 576, 720]
    for size in sizes:
        print(f'Current run: {name}, size={size}.')
        # Read in test images.
        image_paths = [image_path]
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
        search_features = extract_features_global(search_images, size)
        frame_features = extract_features_global(frame_images, size)

        dist = frame_features - search_features[:1]  # for now only consider the first search image
        dist = np.linalg.norm(dist, axis=1)
        res = np.argsort(dist)

        dist = dist[res]

        # calculate the average precision

        img_labels = pd.read_csv(labels_file)
        frame_labels = np.zeros(len(img_labels))
        for my_idx in range(len(img_labels)):
            frame_labels[my_idx] = img_labels.at[my_idx, 'label']
        k = int(sum(frame_labels))  # the number of theoretical hits, assuming every hit is marked with a '1'

        ap = 0
        hits = 0
        for i in range(0, k):
            if frame_labels[res[i]] == 1:
                hits = hits + 1
                ap = ap + hits / (i+1)
        ap = ap / k

        data['name'].append(name)
        data['search_image'].append(os.path.basename(image_path))
        data['num_of_frames'].append(len(frame_features))
        data['ap'].append(ap)
        data['recall'].append(hits / k)
        data['k'].append(k)
        data['size'].append(size)


        print(f'Res has length {len(res)}.\n{res}')
        print(f'Dist:\n{dist}')

    df = pd.DataFrame(data)
    df.to_csv(out_file)


if __name__ == '__main__':
    main()
