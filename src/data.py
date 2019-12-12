# independent file to read the binary file and generate image folder

import os
import sys
import tarfile
import errno
import numpy as np
import matplotlib.pyplot as plt

try:
    from imageio import imsave
except:
    from scipy.misc import imsave


def read_all_images(path):
    with open(path, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        return np.transpose(images, (0, 3, 2, 1))


def save_unlabelled_images(images):
    i = 0
    for image in images:
        dir = 'data/self_supervised/unlabeled/'
        try:
            os.makedirs(dir, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = dir + str(i)
        imsave("%s.png" % filename, image, format="png")
        i += 1


def save_labeled_images(images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        # switch between train and test accordingly
        directory = 'data/supervised/test/' + str(label) + '/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        imsave("%s.png" % filename, image, format="png")
        i = i+1


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


#data_path = './../data/stl10_binary/unlabeled_X.bin'
#images = read_all_images(data_path)
# save_unlabelled_images(images)

#data_path = 'data/stl10_binary/test_X.bin'
#label_path = 'data/stl10_binary/test_y.bin'
#labels = read_labels(label_path)
#images = read_all_images(data_path)
#save_labeled_images(images, labels)
