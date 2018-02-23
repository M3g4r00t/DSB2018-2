import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import misc
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm

# Set some parameters
IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '\\'
TRAIN_PATH = ROOT_PATH + 'input\\stage1_train\\'
TEST_PATH = ROOT_PATH + 'input\\stage1_test\\'
TRAIN_PATH_PP = ROOT_PATH + 'input\\stage1_train_pp_02\\'
TEST_PATH_PP = ROOT_PATH + 'input\\stage1_test_pp_02\\'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed(seed)

# Set enviroment
if not os.path.exists(TRAIN_PATH_PP):
    os.mkdir(TRAIN_PATH_PP)

if not os.path.exists(TEST_PATH_PP):
    os.mkdir(TEST_PATH_PP)

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

print('Getting and resizing train images and masks ... ' + str(len(train_ids)))
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    file = path + '/images/' + id_ + '.png'
    img = imread(file)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    if not os.path.exists(TRAIN_PATH_PP + id_):
        os.mkdir(TRAIN_PATH_PP + id_)
        os.mkdir(TRAIN_PATH_PP + id_ + '/images/')
        os.mkdir(TRAIN_PATH_PP + id_ + '/masks/')
        # imsave(TRAIN_PATH_PP + id_ + '/images/' + id_ + '.png', img)
        misc.imsave(TRAIN_PATH_PP + id_ + '/images/' + id_ + '.png', img)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        plt.imsave(TRAIN_PATH_PP + id_ + '/masks/' + id_ + '.png', np.array(mask).reshape(IMG_HEIGHT, IMG_WIDTH),
                   cmap=cm.gray)

print('Getting and resizing test images ... ' + str(len(test_ids)))
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    file = path + '/images/' + id_ + '.png'
    img = imread(file)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    if not os.path.exists(TEST_PATH_PP + id_):
        os.mkdir(TEST_PATH_PP + id_)
        os.mkdir(TEST_PATH_PP + id_ + '/images/')
        # imsave(TEST_PATH_PP + id_ + '/images/' + id_ + '.png', img)
        misc.imsave(TEST_PATH_PP + id_ + '/images/' + id_ + '.png', img)
