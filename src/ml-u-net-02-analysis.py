import os
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from skimage.io import imread, imshow
from skimage.morphology import label
from tqdm import tqdm

# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
BATCH_SIZE = 10  # the higher the better
ROOT_PATH = 'D:\\Dennys\\Personal\\Cursos\\BecaOEA\\PPGCC\\Others\\Startup\\Kaggler\\DSB2018\\'
TRAIN_PATH = ROOT_PATH + 'input\\stage1_train_pp_02\\'
TEST_PATH = ROOT_PATH + 'input\\stage1_test\\'
MODEL = 'model-dsbowl2018-4'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
input_seed = 42
random.seed = input_seed
np.random.seed(input_seed)

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

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

# Get and resize test images
'''
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ' + str(len(test_ids)))
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
'''
print('Done!')


# Define IoU metric
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


# Predict on train, val and test
model = load_model(ROOT_PATH + 'output\\' + MODEL + '.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train, verbose=1)
'''
preds_test = model.predict(X_test, verbose=1)
'''

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
'''
preds_test_t = (preds_test > 0.5).astype(np.uint8)
'''
# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()


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


def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        true_negatives = np.sum(matches, axis=0) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, tn, fp, fn = np.sum(true_positives), np.sum(true_negatives), np.sum(false_positives), np.sum(false_negatives)
        return tp, tn, fp, fn

    # Loop over IoU thresholds
    prec = []
    tp_array = []
    tn_array = []
    fp_array = []
    fn_array = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, tn, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
        tp_array.append(tp)
        tn_array.append(tn)
        fp_array.append(fp)
        fn_array.append(fn)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec), np.mean(tp_array), np.mean(tn_array), np.mean(fp_array), np.mean(fn_array)


def mean_iou_array(y_true_array, y_pred_array):
    prec = []
    tp_array = []
    tn_array = []
    fp_array = []
    fn_array = []
    print('Getting train scores ... ')
    sys.stdout.flush()
    for n, _ in tqdm(enumerate(y_true_array), total=len(y_true_array)):
        score, tp, tn, fp, fn = iou_metric(y_true_array[n], y_pred_array[n])
        prec.append(score)
        tp_array.append(tp)
        tn_array.append(tn)
        fp_array.append(fp)
        fn_array.append(fn)
    return prec, tp_array, tn_array, fp_array, fn_array


sub = pd.DataFrame()
sub['ImageId'] = train_ids
score, tp, tn, fp, fn = mean_iou_array(Y_train, preds_train_t)
sub['IoU'] = score
sub['tp'] = tp
sub['tn'] = tn
sub['fp'] = fp
sub['fn'] = fn
sub.to_csv(ROOT_PATH + 'output\\' + MODEL + '-analysis.csv', index=False)
