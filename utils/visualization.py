'''
Tools / utilities / helper methods pertaining to qualitative deep dives into train / test results.
Created by Basile Van Hoorick.
'''

import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'eval/'))
sys.path.insert(0, os.getcwd())

from __init__ import *


def draw_text(image, x, y, label, color, size_mult=1.0):
    '''
    :param image (H, W, 3) array of float32 in [0, 1].
    :param x (int): x coordinate of the top left corner of the text.
    :param y (int): y coordinate of the top left corner of the text.
    :param label (str): Text to draw.
    :param color (3) tuple of float32 in [0, 1]: RGB values.
    :param size_mult (float): Multiplier for font size.
    :return image (H, W, 3) array of float32 in [0, 1].
    '''
    # Draw background and write text using OpenCV.
    label_width = int((16 + len(label) * 10) * size_mult)
    label_height = int(22 * size_mult)
    image[y:y + label_height, x:x + label_width] = (0, 0, 0)
    image = cv2.putText(image, label, (x, y + label_height - 8), 2,
                        0.5 * size_mult, color, thickness=int(size_mult))
    return image


def quick_pca(array, k=3, unique_features=False, normalize=None):
    '''
    array (*, n): Array to perform PCA on.
    k (int) < n: Number of components to keep.
    '''
    n = array.shape[-1]
    all_axes_except_last = tuple(range(len(array.shape) - 1))
    array_flat = array.reshape(-1, n)

    pca = sklearn.decomposition.PCA(n_components=k)

    if unique_features:
        # Obtain unique combinations of occluding instance sequences, to avoid bias toward larger
        # object masks.
        unique_combinations = np.unique(array_flat, axis=0)
        pca.fit(unique_combinations)

    else:
        pca.fit(array_flat)

    result_unnorm = pca.transform(array_flat).reshape(*array.shape[:-1], k)

    if normalize is not None:
        per_channel_min = result_unnorm.min(axis=all_axes_except_last, keepdims=True)
        per_channel_max = result_unnorm.max(axis=all_axes_except_last, keepdims=True)
        result = (result_unnorm - per_channel_min) / (per_channel_max - per_channel_min)
        result = result * (normalize[1] - normalize[0]) + normalize[0]

    else:
        result = result_unnorm

    result = result.astype(np.float32)
    return result
