import os
import random
import sys
import warnings

import numpy as np
from imgaug import augmenters as iaa, misc
from skimage.io import imread
from tqdm import tqdm
from skimage.transform import resize

# Set some parameters
IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')) + '\\'
TRAIN_PATH = ROOT_PATH + 'input\\stage1_train\\'
TEST_PATH = ROOT_PATH + 'input\\stage1_test\\'
TRAIN_PATH_PP = ROOT_PATH + 'input\\stage1_train_pp_02\\'
TEST_PATH_PP = ROOT_PATH + 'input\\stage1_test_pp_02\\'
TRAIN_PATH_AUG = ROOT_PATH + 'input\\stage1_train_aug_02\\'
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
input_seed = 42
random.seed = seed
np.random.seed(seed)
BATCH_SIZE = 32

# Set enviroment
if not os.path.exists(TRAIN_PATH_AUG):
    os.mkdir(TRAIN_PATH_AUG)

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH_PP))[1]

print('Getting and transforming train images and masks ... ' + str(len(train_ids)))
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH_PP + id_
    file = path + '/images/' + id_ + '.png'
    img = imread(file)[:, :, :IMG_CHANNELS]

    # Transform #0

    id_aug = id_ + '-0'
    seq = iaa.Sequential([iaa.Fliplr(1, deterministic=True),
                          iaa.Fliplr(1, deterministic=True)], deterministic=True)  # none

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #1

    id_aug = id_ + '-1'
    seq = iaa.Sequential([iaa.Fliplr(1, deterministic=True)], deterministic=True)  # horizontal flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #2

    id_aug = id_ + '-2'
    seq = iaa.Sequential([iaa.Flipud(1, deterministic=True)], deterministic=True)  # vertical flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #3

    id_aug = id_ + '-3'
    seq = iaa.Sequential([iaa.Fliplr(1, deterministic=True), iaa.Flipud(1, deterministic=True)],
                         deterministic=True)  # horizontal + vertical flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #4

    id_aug = id_ + '-4'
    seq = iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                          iaa.WithChannels([0, 1, 2], iaa.Add((50, 100))),
                          iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")])  # change color

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = mask
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #5

    id_aug = id_ + '-5'
    seq = iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                          iaa.WithChannels([0, 1, 2], iaa.Add((50, 100))),
                          iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                          iaa.Fliplr(1, deterministic=True)])  # change color + horizontal flip

    seq_mask = iaa.Sequential([iaa.Fliplr(1, deterministic=True)], deterministic=True)  # horizontal flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq_mask.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #6

    id_aug = id_ + '-6'
    seq = iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                          iaa.WithChannels([0, 1, 2], iaa.Add((50, 100))),
                          iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                          iaa.Flipud(1, deterministic=True)])  # change color + vertical flip

    seq_mask = iaa.Sequential([iaa.Flipud(1, deterministic=True)], deterministic=True)  # vertical flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq_mask.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)

    # Transform #7

    id_aug = id_ + '-7'
    seq = iaa.Sequential([iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                          iaa.WithChannels([0, 1, 2], iaa.Add((50, 100))),
                          iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB"),
                          iaa.Fliplr(1, deterministic=True),
                          iaa.Flipud(1, deterministic=True)])  # change color + horizontal + vertical flip

    seq_mask = iaa.Sequential([iaa.Fliplr(1, deterministic=True), iaa.Flipud(1, deterministic=True)],
                              deterministic=True)  # horizontal + vertical flips

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = seq.augment_image(img)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = imread(path + '/masks/' + id_ + '.png')
        mask = np.maximum(mask, mask_)
        mask_aug = seq_mask.augment_image(mask)
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)


# Load input data
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    X_train[n] = img
    mask = imread(path + '/masks/' + id_ + '.png')[:, :, :1]
    Y_train[n] = mask


from keras.preprocessing import image

# Creating the training Image and Mask generator
image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                         height_shift_range=0.2, fill_mode='reflect')
mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                        height_shift_range=0.2, fill_mode='reflect')

# Keep the same seed for image and mask generators so they fit together

image_datagen.fit(X_train, augment=True, seed=input_seed)
mask_datagen.fit(Y_train, augment=True, seed=input_seed)

x = image_datagen.flow(X_train, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)
y = mask_datagen.flow(Y_train, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH_PP + id_
    file = path + '/images/' + id_ + '.png'
    img = imread(file)[:, :, :IMG_CHANNELS]

    # Transform #0

    id_aug = id_ + '-8'

    if not os.path.exists(TRAIN_PATH_AUG + id_aug):
        os.mkdir(TRAIN_PATH_AUG + id_aug)
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/images/')
        os.mkdir(TRAIN_PATH_AUG + id_aug + '/masks/')
        img_aug = x[n]
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/images/' + id_aug + '.png', img_aug)
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        mask_ = y[n]
        mask = np.maximum(mask, mask_)
        mask_aug = mask
        misc.imsave(TRAIN_PATH_AUG + id_aug + '/masks/' + id_aug + '.png', mask_aug)


