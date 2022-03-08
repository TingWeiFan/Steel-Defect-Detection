import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2, csv, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy

from modules import UnetPP_Model
from data_generator import DataGenerator
from predict import Prediction

import warnings
warnings.filterwarnings('ignore')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
session = tf.Session(config=config)


def tversky_coef(y_true, y_pred):
    p0, g0 = y_pred, y_true
    p1, g1 = 1-y_pred, 1-y_true
    alpha = 0.3
    beta = 0.7

    num = k.sum(p0 * g0, axis=(1, 2, 3))
    den = num + alpha * k.sum(p0 * g1, axis=(1, 2, 3)) + beta * k.sum(p1 * g0, axis=(1, 2, 3))
    T = num / den

    dices = k.mean(T, axis=0)
    return k.mean(dices)

def tversky_coef_loss(y_true, y_pred):
    return 1 - tversky_coef(y_true, y_pred)

def dice_coefficient(y_true, y_pred):
    sum1 = 2 * tf.math.reduce_sum(y_true * y_pred, axis=(0, 1, 2))
    sum2 = tf.math.reduce_sum(y_true**2 + y_pred**2, axis=(0, 1, 2))
    dice = sum1 / (sum2 + 1e-9)
    dice = tf.math.reduce_mean(dice)
    return dice

def dice_loss(y_true, y_pred):
    return (1-dice_coefficient(y_true, y_pred))

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1-dice_coefficient(y_true, y_pred))

#---------------------------------------------------------------#
full_data    = pd.read_csv('./data/unique_rle_1234.csv').fillna('')
train_data   = pd.read_csv('./data/train_rle_1234.csv').fillna('')
validtn_data = pd.read_csv('./data/val_rle_1234.csv').fillna('')
test_data    = pd.read_csv('./data/test_rle_1234.csv').fillna('')

#---------------------------------------------------------------#
input_img = Input((256, 1600, 3), name='img')
model = UnetPP_Model(input_img, n_filters=16, dropout=0.1, batchnorm=True)
#model.compile(optimizer=Adam(0.02), loss=tversky_coef_loss, metrics=[tversky_coef])
model.compile(optimizer=Adam(), loss=dice_loss, metrics=[dice_coefficient])

train_batches = DataGenerator(train_data, shuffle=True)
validtn_batches = DataGenerator(validtn_data, shuffle=False)

checkpoints = ModelCheckpoint('unet++.h5', monitor='val_dice_coefficient', verbose=1, save_best_only=True, mode='max') #val_tversky_coef
history = model.fit(train_batches, validation_data=validtn_batches, epochs=60, callbacks=[checkpoints])

#---------------------------------------------------------------#
model_path = './'
model.load_weights('./result/m22/unet++.h5')
mp = Prediction(full_data, model)

for i in ['e75c7bbd9', 'f9b98ab64', 'f4dde5dac']:
    _, _ = mp.visualize_model_prediction(model_path, '{}.jpg'.format(i), show=True)

for i in ['d26ead1d8_flip2', 'cd1f8d368_flip2', 'dc636ab48_flip3']:
    _, _ = mp.visualize_model_prediction(model_path, '{}.jpg'.format(i), show=True)

for i in ['2a351164a', '2a74520ed', '2c061aed6']:
    _, _ = mp.visualize_model_prediction(model_path, '{}.jpg'.format(i), show=True)

for i in ['ff9d46e95', 'fd26ab9ad', 'fbadf780f']:
    _, _ = mp.visualize_model_prediction(model_path, '{}.jpg'.format(i), show=True)


with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'class_id', 'ground_truth', 'pred'])

    test_name = list(test_data['ImageId'])
    loss = []
    for i in range(len(test_name)):
        img_acc, pred = mp.visualize_model_prediction(model_path, '{}'.format(test_name[i]), show=False, save=True)
        for n in range(4):
            res = pred[0, :, :, n].reshape((1, -1))
            detect = False if np.count_nonzero(res[0] == 0.0) == 256 * 1600 else True
            ans = test_data.iloc[i][n+1]
            gt = False if ans == '' else True
            writer.writerow([test_name[i], n+1, gt, detect])
        loss.append(img_acc)

    print(sum(loss)/len(loss))