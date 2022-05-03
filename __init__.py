'''
These imports are shared across all files.
'''

# Library imports.
import argparse
import collections
import collections.abc
import copy
import cv2
import imageio
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pathlib
import pickle
import platform
import random
import scipy
import seaborn as sns
import shutil
import sklearn
import sklearn.decomposition
import sys
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.io
import torchvision.models
import torchvision.transforms
import torchvision.utils
import tqdm
import warnings
from einops import rearrange, repeat

PROJECT_NAME = 'my-project-template'

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'data/'))
sys.path.append(os.path.join(os.getcwd(), 'eval/'))
sys.path.append(os.path.join(os.getcwd(), 'model/'))
sys.path.append(os.path.join(os.getcwd(), 'utils/'))
