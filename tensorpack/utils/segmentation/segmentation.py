#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: segmentation.py
# Author: Tao Hu <taohu620@gmail.com>

import numpy as np
from math import ceil
import cv2,colorsys
import matplotlib.pyplot as plt
#import pydensecrf.densecrf as dcrf
import os, sys
#from ...utils import logger


__all__ = ['update_confusion_matrix', 'predict_slider']

# Colour map.
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

ignore_color = (255,255,255)
fuzzy_color = (64,0,128)


def update_confusion_matrix(pred, label, conf_m, nb_classes, ignore = 255):

        ignore_index = label != ignore
        seg_gt = label[ignore_index].astype('int32')
        seg_pred = pred[ignore_index].astype('int32')
        index = (seg_gt * nb_classes + seg_pred).astype('int32')
        label_count = np.bincount(index)
        for i_label in range(nb_classes):
            for i_pred_label in range(nb_classes):
                cur_index = i_label * nb_classes + i_pred_label
                if cur_index < len(label_count):
                    conf_m[i_label, i_pred_label] += label_count[cur_index] #notice here, first dimension is label,second dimension is prediction.
        return conf_m


def imwrite_grid(image,label,prediction,border,prefix_dir, imageId):
    h,w,_ = image.shape
    grid_num = h/border
    for i in range(grid_num):
        for j in range(grid_num):
            start_i = border*i
            start_j = border*j
            end_i = border*(i+1)
            end_j = border*(j+1)
            cv2.imwrite(os.path.join(prefix_dir,"out{}_patch{}_{}.png".format(imageId,i,j)),
                        np.concatenate((image[start_i:end_i,start_j:end_j],
                                        visualize_label(label[start_i:end_i,start_j:end_j]),
                                        visualize_label(prediction[start_i:end_i,start_j:end_j]*2)), axis=1))


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[0], 0)
    cols_missing = max(target_size[1] - img.shape[1], 0)
    try:
        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    except  Exception as e:
        print(str(e))
        pass
    return padded_img, [0,target_size[0]-rows_missing,0,target_size[1] - cols_missing]


def pad_edge(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = max(target_size[0] - img.shape[0], 0)
    cols_missing = max(target_size[1] - img.shape[1], 0)
    padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, 0)), 'constant')
    return padded_img, [0,target_size[0]-rows_missing,0,target_size[1] - cols_missing]


def visualize_label(label):
    """Color classes a good distance away from each other."""
    h, w = label.shape
    img_color = np.zeros((h, w, 3)).astype('uint8')
    for i in range(0,21):
        img_color[label == i] = label_colours[i]

    img_color[label==255] = ignore_color#highlight ignore label
    return img_color


def visualize_mixlabel(label,mask):#H,W,C
    """Color classes a good distance away from each other."""
    h, w, c = label.shape
    img_color = np.zeros((h, w, 3)).astype('uint8')
    for i in range(h):
        for j in range(w):
            aa = 0
            bb=0
            cc=0
            if len(np.where(label[i,j]>0)[0]) >1:
                img_color[i,j] = fuzzy_color
                continue

            for k in range(c):
                if label[i, j, k] > 0:
                    aa += label[i, j, k]*label_colours[k][0]
                    bb + label[i, j, k]*label_colours[k][1]
                    cc += label[i, j, k]*label_colours[k][2]

            img_color[i,j] = [aa,bb,cc]


    img_color[mask==0] = ignore_color#highlight ignore label
    return img_color


def predict_slider(full_image, predictor, classes, tile_size):
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    overlap = 1/3
    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows
            img = full_image[y1:y2, x1:x2]
            padded_img, padding_index = pad_image(img, tile_size) #only happen in validation or test when the original image size is already smaller than tile_size
            tile_counter += 1
            padded_img = padded_img[None, :, :, :].astype('float32') # extend one dimension
            padded_prediction = predictor(padded_img)[0][0]
            prediction_no_padding = padded_prediction[padding_index[0]:padding_index[1],padding_index[2]:padding_index[3],:]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction_no_padding  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs

def predict_scaler(full_image, predictor, scales, classes, tile_size, is_densecrf):
    """scaler is only respnsible for generate multi scale input for slider"""
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    h_ori, w_ori = full_image.shape[:2]
    for scale in scales:
        scaled_img = cv2.resize(full_image, (int(scale*w_ori), int(scale*h_ori)))
        scaled_probs = predict_slider(scaled_img, predictor, classes, tile_size)
        probs = cv2.resize(scaled_probs, (w_ori,h_ori))
        full_probs += probs
    full_probs /= len(scales)
    if is_densecrf:
        full_probs = dense_crf(full_probs)
    return full_probs



def edge_predict_slider(full_image, edge, predictor, classes, tile_size):
    """slider is responsible for generate slide window,
    the window image may be smaller than the original image(if the original image is smaller than tile_size),
     so we need to padding.
     here we should notice that the network input is fixed.
     before send the image into the network, we should do some padding"""
    tile_size = (tile_size, tile_size)
    overlap = 1/3
    stride = ceil(tile_size[0] * (1 - overlap))
    tile_rows = int(ceil((full_image.shape[0] - tile_size[0]) / stride) + 1)  # strided convolution formula
    tile_cols = int(ceil((full_image.shape[1] - tile_size[1]) / stride) + 1)
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    count_predictions = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    tile_counter = 0
    for row in range(tile_rows):
        for col in range(tile_cols):
            x1 = int(col * stride)
            y1 = int(row * stride)
            x2 = min(x1 + tile_size[1], full_image.shape[1])
            y2 = min(y1 + tile_size[0], full_image.shape[0])
            x1 = max(int(x2 - tile_size[1]), 0)  # for portrait images the x1 underflows sometimes
            y1 = max(int(y2 - tile_size[0]), 0)  # for very few rows y1 underflows
            img = full_image[y1:y2, x1:x2]
            ed = edge[y1:y2, x1:x2]

            padded_img, padding_index = pad_image(img, tile_size) #only happen in validation or test when the original image size is already smaller than tile_size
            padded_ed, padding_ed_index = pad_image(ed, tile_size)

            tile_counter += 1
            padded_img = padded_img[None, :, :, :].astype('float32') # extend one dimension
            padded_ed = np.squeeze(padded_ed)
            padded_ed = padded_ed[None, :, :].astype('float32')  # extend one dimension

            padded_prediction = predictor(padded_img, padded_ed)[0][0]
            prediction_no_padding = padded_prediction[padding_index[0]:padding_index[1],padding_index[2]:padding_index[3],:]
            count_predictions[y1:y2, x1:x2] += 1
            full_probs[y1:y2, x1:x2] += prediction_no_padding  # accumulate the predictions also in the overlapping regions

    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs


def edge_predict_scaler(full_image, edge, predictor, scales, classes, tile_size, is_densecrf):
    """scaler is only respnsible for generate multi scale input for slider"""
    full_probs = np.zeros((full_image.shape[0], full_image.shape[1], classes))
    h_ori, w_ori = full_image.shape[:2]
    for scale in scales:
        scaled_img = cv2.resize(full_image, (int(scale*w_ori), int(scale*h_ori)))
        scaled_edge = cv2.resize(edge, (int(scale * w_ori), int(scale * h_ori)))#resize on single channel will make extra dim disappear!!!
        scaled_probs = edge_predict_slider(scaled_img, scaled_edge[:,:,None], predictor, classes, tile_size)
        probs = cv2.resize(scaled_probs, (w_ori,h_ori))
        full_probs += probs
    full_probs /= len(scales)
    #if is_densecrf:
    #    full_probs = dense_crf(full_probs)
    return full_probs


if __name__ == '__main__':
    label = np.array([1,1,1,0,0,0,0])
    pred = np.array([0,0,0,0,0,0,0])
    cm = np.array([[0,0],[0,0]])
    cm = update_confusion_matrix(pred,label,cm,2)
    pass







