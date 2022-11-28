'''
Tools / utilities / helper methods pertaining to qualitative deep dives into train / test results.
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
