import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from skimage.io import imread, imshow, imsave
from skimage.morphology import label
from skimage.transform import resize
from sklearn import model_selection
from tqdm import tqdm

# Image dimensions (resize)
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

# Batch size for weight calculation
BATCH_SIZE = 16

# Filename setup
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + '\\'
TRAIN_PATH = ROOT_PATH + 'input\\stage1_train_aug_02\\'
TEST_PATH = ROOT_PATH + 'input\\stage1_test\\'
MODEL_FILE_NAME = 'model-dsbowl2018-4.h5'
MODEL_FILE_PATH = ROOT_PATH + 'output\\' + MODEL_FILE_NAME
VIEW_IMAGE_FLAG = True
SAVE_IMAGE_RESULTS = True

# Skimage warnings
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# For experiment replication
input_seed = 42  # the universe response
random.seed = input_seed
np.random.seed(input_seed)

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Load input data
# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
''' Begin: Erase in production '''
X_train = X_train[:32 * 4]
Y_train = Y_train[:32 * 4]
train_ids = train_ids[:32 * 4]
''' End: Erase in production '''
print('Getting train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    X_train[n] = img
    mask = imread(path + '/masks/' + id_ + '.png')[:, :, :1]
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting test images ... ' + str(len(test_ids)))
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X_train, Y_train, test_size=0.25,
                                                                  random_state=input_seed)

if VIEW_IMAGE_FLAG:
    try:
        # Check if training data looks all right
        ix = random.randint(0, len(X_train))
        imshow(X_train[ix])
        plt.show()
        imshow(np.squeeze(Y_train[ix]))
        plt.show()
        ix = random.randint(0, len(X_val))
        imshow(X_val[ix])
        plt.show()
        imshow(np.squeeze(Y_val[ix]))
        plt.show()
    except Exception as e:
        print("Unexpected error:", e.__str__())


# Define IoU metric: IoU = (A Intersection B)/(A union B)
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


if not os.path.exists(MODEL_FILE_PATH):
    # Data augmentation

    print('Data augmentation for validation ...')

    # Creating the training Image and Mask generator
    image_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                             height_shift_range=0.2, fill_mode='reflect')
    mask_datagen = image.ImageDataGenerator(shear_range=0.5, rotation_range=50, zoom_range=0.2, width_shift_range=0.2,
                                            height_shift_range=0.2, fill_mode='reflect')

    # Keep the same seed for image and mask generators so they fit together

    image_datagen.fit(X_train, augment=True, rounds=3, seed=input_seed)
    mask_datagen.fit(Y_train, augment=True, rounds=3, seed=input_seed)

    x = image_datagen.flow(X_train, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)
    y = mask_datagen.flow(Y_train, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)

    # Creating the validation Image and Mask generator
    image_datagen_val = image.ImageDataGenerator()
    mask_datagen_val = image.ImageDataGenerator()

    image_datagen_val.fit(X_val, augment=True, seed=input_seed)
    mask_datagen_val.fit(Y_val, augment=True, seed=input_seed)

    x_val = image_datagen_val.flow(X_val, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)
    y_val = mask_datagen_val.flow(Y_val, batch_size=BATCH_SIZE, shuffle=True, seed=input_seed)

    if VIEW_IMAGE_FLAG:
        try:
            # Checking if the images fit
            imshow(x.next()[0].astype(np.uint8))
            plt.show()
            imshow(np.squeeze(y.next()[0].astype(np.uint8)))
            plt.show()
            imshow(x_val.next()[0].astype(np.uint8))
            plt.show()
            imshow(np.squeeze(y_val.next()[0].astype(np.uint8)))
            plt.show()
        except Exception as e:
            print("Unexpected error:", e.__str__())

    # creating a training and validation generator that generate masks and images
    train_generator = zip(x, y)
    val_generator = zip(x_val, y_val)

    print('Build network ...')

    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    # Fit model
    earlystopper = EarlyStopping(patience=10, verbose=1, mode='min')
    checkpointer = ModelCheckpoint(MODEL_FILE_PATH, verbose=1, save_best_only=True)

    print('Train network')
    results = model.fit_generator(train_generator, validation_data=val_generator, validation_steps=BATCH_SIZE,
                                  steps_per_epoch=250,
                                  epochs=200, callbacks=[earlystopper, checkpointer])

print('Reading model > ' + MODEL_FILE_PATH)
# Predict on train, val and test
model = load_model(MODEL_FILE_PATH, custom_objects={'mean_iou': mean_iou})
print(model.summary())
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_val, verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

if VIEW_IMAGE_FLAG:
    try:
        # Perform a sanity check on some random training samples
        ix = random.randint(0, len(preds_train_t))
        imshow(X_train[ix])
        plt.show()
        imshow(np.squeeze(Y_train[ix]))
        plt.show()
        imshow(np.squeeze(preds_train_t[ix].astype(np.uint8)))
        plt.show()

        # Perform a sanity check on some random validation samples
        ix = random.randint(0, len(preds_val_t))
        imshow(X_val[ix])
        plt.show()
        imshow(np.squeeze(Y_val[ix].astype(np.uint8)))
        plt.show()
        imshow(np.squeeze(preds_val_t[ix].astype(np.uint8)))
        plt.show()

        # Perform a sanity check on some random validation samples
        ix = random.randint(0, len(preds_test_t))
        imshow(X_test[ix])
        plt.show()
        imshow(np.squeeze(preds_test_t[ix].astype(np.uint8)))
        plt.show()

    except Exception as e:
        print("Unexpected error:", e.__str__())


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for ii in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == ii)


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

print('Create submission DataFrame ...')

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv(ROOT_PATH + 'output\\sub-dsbowl2018-' + MODEL_FILE_NAME + '-20180319.csv', index=False)

if SAVE_IMAGE_RESULTS:
    for n, id_ in enumerate(test_ids):
        path = TEST_PATH + id_
        imsave(path + '/images/' + MODEL_FILE_NAME + '.png', resize(np.squeeze(preds_test_t[n]),
                                                                    (sizes_test[n][0], sizes_test[n][1]),
                                                                    mode='constant', preserve_range=True))
