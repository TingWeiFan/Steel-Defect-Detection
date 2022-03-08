import os
from PIL import Image
import numpy as np
from collections import defaultdict
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import csv


train_df = pd.read_csv("./data/train.csv")

l1, l2, l3, l4 = [], [], [], []

for col in range(len(train_df)):
    img_names = train_df["ImageId"][col]
    labels = train_df["ClassId"][col]
    if labels == 1:
        l1.append(col)
    elif labels == 2:
        l2.append(col)
    elif labels == 3:
        l3.append(col)
    elif labels == 4:
        l4.append(col)

print("type 1:{}, type 2:{}, type 3:{}, type 4:{}".format(len(l1), len(l2), len(l3), len(l4)))

#--------------------------------------------------------------------------------------------------#

palet = [(249, 192, 12), (0, 185, 241), (114, 0, 218), (249,50,12)]

def name_and_mask(start_idx):
    col = start_idx
    img_names = train_df["ImageId"][col]
    labels = train_df["ClassId"][col]
    pixels = train_df["EncodedPixels"][col]
    mask = np.zeros((256, 1600, 4), dtype=np.uint8)

    for idx, label in enumerate([pixels]):
        mask_label = np.zeros(1600*256, dtype=np.uint8)
        
        label = label.split(" ")
        positions = map(int, label[0::2]) # start pixel
        length = map(int, label[1::2])    # continue length
        
        for pos, le in zip(positions, length):
            mask_label[pos-1:pos+le-1] = 1

        mask[:, :, idx] = mask_label.reshape(256, 1600, order='F')

    return img_names, mask

def show_mask_image(col):
    name, mask = name_and_mask(col)
    img = cv2.imread("./data/train_images/{}".format(name))

    for ch in range(4):
        #contours, _ = cv2.findContours(mask[:, :, ch], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)                  # for opencv 2.4.x
        _,contours,_ = cv2.findContours(mask[:, :, ch].astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # for opencv 3
    
        for i in range(0, len(contours)):
            cv2.polylines(img, contours[i], True, palet[ch], 2)
    
    return img, mask

def mask_process(mask):
    crop_img = mask.reshape(1, 256*1600*4, order='F')
    pixels = []
    last_pixel = None
    continues = 0
    for row in range(len(crop_img[0])):
        if crop_img[0][row] == 1 and continues == 0:
            pixels.append(row)
            last_pixel = int(row)
            continues += 1
        elif crop_img[0][row] == 1 and continues > 0 and row - last_pixel == 1:
            last_pixel = int(row)
            continues += 1
        elif crop_img[0][row] == 1 and continues > 0 and row - last_pixel != 1:
            pixels.append(continues)
            pixels.append(row)
            last_pixel = int(row)
            continues = 1

    pixels.append(continues)
    text = ''
    for e in range(len(pixels)):
        if e == 0:
            text = str(pixels[e]) + ' '
        elif e < len(pixels):
            text = text + str(pixels[e]) + ' '
        else:
            text = text + str(pixels[e])

    return text

new_data = pd.read_csv('./data/aug_train.csv').fillna('')
cls1, cls2, cls3, cls4 = 0, 0, 0, 0
classes = [cls1, cls2, cls3, cls4]
for i in range(len(new_data)):
    if list(new_data.loc[i])[1] == 1: cls1 += 1

    elif list(new_data.loc[i])[1] == 2: cls2 += 1

    elif list(new_data.loc[i])[1] == 3: cls3 += 1

    elif list(new_data.loc[i])[1] == 4: cls4 += 1

print('class 1:{}, class 2:{}, class 3:{}, class 4:{}'.format(cls1, cls2, cls3, cls4))