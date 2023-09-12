import time
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath('./solar/solar_global')))
from fe_main import extract_features_global


def plot_data(data):
    cm = 1/2.54  # convert inches to centimeters
    fig, ax = plt.subplots(figsize=(18*cm, 18*cm))
    ax.plot(data['h'], data['time'])

    # set figure data
    ax.set_title(f'Time vs. Resolution')
    ax.set_xlabel('Height [px]')
    ax.set_ylabel('Time [s]')
    ax.set_xticks(data['h'])
    ax.set_yticks(list(round(x, 2) for x in data['time']))
    plt.xlim([min(data['h']), max(data['h'])])
    plt.ylim([min(data['time']), max(data['time'])])
    plt.grid()

    plt.show()


def main():
    """Main.

    Use 16:9 aspect ratio. WxH
    """
    n = 50  # the more iterations the better, because then the loading time for the network disappears
    size = ((123, 33), (256, 144), (320, 180), (426, 240), (640, 360), (848, 480), (960, 540), (1024, 576),
            (1280, 720))  # we add one bogus iteration at the start to eliminate the initial long loading time

    data = {
        'w': [],
        'h': [],
        'time': []
    }

    for s in size:
        # generate some random images
        images = (np.random.rand(*s, 3) * 255 for i in range(n))
        images = list(Image.fromarray(x.astype('uint8')).convert('RGB') for x in images)

        start_time = time.time()  # time.process_time() returned weird results, time.time() is more reliable
        _ = extract_features_global(images, s[1])
        end_time = time.time()

        data['w'].append(s[0])
        data['h'].append(s[1])
        data['time'].append((end_time - start_time)/n)

    data['w'].pop(0)  # remove the bogus iteration
    data['h'].pop(0)
    data['time'].pop(0)
    plot_data(data, n)
    print(data)


if __name__ == '__main__':
    main()
