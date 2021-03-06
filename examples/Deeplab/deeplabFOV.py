#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: deeplabFOV.py
# Author: Tao Hu <taohu620@gmail.com>

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os
import numpy as np

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.segmentation import predict_slider, visualize_label
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
import tensorpack.tfutils.symbolic_functions as symbf
from tqdm import tqdm



CLASS_NUM = 21
CROP_SIZE = 321
IGNORE_LABEL = 255

class Model(ModelDesc):

    def _get_inputs(self):
        ## Set static shape so that tensorflow knows shape at compile time.
        return [InputDesc(tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'image'),
                InputDesc(tf.int32, [None, CROP_SIZE, CROP_SIZE], 'gt')]

    def _build_graph(self, inputs):
        def vgg16(image):
            with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
                def aspp_branch(input, rate):
                    input = AtrousConv2D('aspp{}_conv0'.format(rate), input, 1024, kernel_shape=3, rate=rate, nl=tf.nn.relu)
                    input = Dropout('aspp{}_dropout0'.format(rate), input, 0.5)
                    input = Conv2D('aspp{}_conv1'.format(rate), input, 1024, kernel_shape=1)
                    input = Dropout('aspp{}_dropout1'.format(rate), input, 0.5)
                    input = Conv2D('aspp{}_conv2'.format(rate), input, CLASS_NUM, kernel_shape=1, nl=tf.identity)
                    return input

                l = Conv2D('conv1_1', image, 64)
                l = Conv2D('conv1_2', l, 64)
                l = MaxPooling('pool1', l, shape=3, stride=2)
                # 112
                l = Conv2D('conv2_1', l, 128)
                l = Conv2D('conv2_2', l, 128)
                l = MaxPooling('pool2', l, shape=3, stride=2)
                # 56
                l = Conv2D('conv3_1', l, 256)
                l = Conv2D('conv3_2', l, 256)
                l = Conv2D('conv3_3', l, 256)
                l = MaxPooling('pool3', l, shape=3, stride=2)
                # 28
                l = Conv2D('conv4_1', l, 512)
                l = Conv2D('conv4_2', l, 512)
                l = Conv2D('conv4_3', l, 512)
                l = MaxPooling('pool4', l, shape=3, stride=1)  # original VGG16 pooling is 2, here is 1
                # 28
                l = AtrousConv2D('conv5_1', l, 512, kernel_shape=3, rate=2)
                l = AtrousConv2D('conv5_2', l, 512, kernel_shape=3, rate=2)
                l = AtrousConv2D('conv5_3', l, 512, kernel_shape=3, rate=2)
                l = MaxPooling('pool5', l, shape=3, stride=1)
                # 28
                #dilation6 = aspp_branch(l, rate=6)
                dilation12 = aspp_branch(l, rate=12)
                #dilation18 = aspp_branch(l, rate=18)
                #dilation24 = aspp_branch(l, rate=24)
                #predict = dilation6 + dilation12 + dilation18 + dilation24
                predict = dilation12
                predict = tf.image.resize_bilinear(predict, image.shape[1:3])

                return predict

        image, label = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        label = tf.identity(label, name="label")

        predict = vgg16(image)

        costs = []
        prob = tf.identity(predict, name='prob')

        label4d = tf.expand_dims(label, 3, name='label4d')
        new_size = prob.get_shape()[1:3]
        label_resized = tf.image.resize_nearest_neighbor(label4d, new_size)

        cost = symbf.softmax_cross_entropy_with_ignore_label(logits=prob, label=label_resized,
                                                             class_num=CLASS_NUM)
        prediction = tf.argmax(prob, axis=-1,name="prediction")
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')  # the average cross-entropy loss
        costs.append(cost)

        if get_current_tower_context().is_training:
            wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                              80000, 0.7, True)
            wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
            costs.append(wd_cost)

            add_param_summary(('.*/W', ['histogram']))   # monitor W
            self.cost = tf.add_n(costs, name='cost')
            add_moving_summary(costs + [self.cost])

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2.5e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('aspp.*_conv.*/W', 10),('aspp.*_conv.*/b', 20), ('conv.*/b', 2)])])


def get_data(name, data_dir, meta_dir, batch_size):
    isTrain = name == 'train'
    ds = dataset.PascalVOC12(data_dir, meta_dir, name, shuffle=True)

    class RandomCropWithPadding(imgaug.ImageAugmentor):
        def _get_augment_params(self, img):
            self.h0 = img.shape[0]
            self.w0 = img.shape[1]

            if CROP_SIZE > self.h0:
                top = (CROP_SIZE - self.h0) / 2
                bottom = (CROP_SIZE - self.h0) - top
            else:
                top = 0
                bottom = 0

            if CROP_SIZE > self.w0:
                left = (CROP_SIZE - self.w0) / 2
                right = (CROP_SIZE - self.w0) - left
            else:
                left = 0
                right = 0
            new_shape = (top + bottom + self.h0, left + right + self.w0)
            diffh = new_shape[0] - CROP_SIZE
            assert diffh >= 0
            crop_start_h = 0 if diffh == 0 else self.rng.randint(diffh)
            diffw = new_shape[1] - CROP_SIZE
            assert diffw >= 0
            crop_start_w = 0 if diffw == 0 else self.rng.randint(diffw)
            return (top, bottom, left, right, crop_start_h, crop_start_w)

        def _augment(self, img, param):
            top, bottom, left, right, crop_start_h, crop_start_w = param
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=IGNORE_LABEL)
            assert crop_start_h + CROP_SIZE <= img.shape[0], crop_start_w + CROP_SIZE <= img.shape[1]
            return img[crop_start_h:crop_start_h + CROP_SIZE, crop_start_w:crop_start_w + CROP_SIZE]


    if isTrain:
        shape_aug = [
            RandomCropWithPadding(),
            imgaug.Flip(horiz=True),
        ]
    else:
        shape_aug = []
        pass
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)


    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data(data_dir, meta_dir, batch_size):
    ds = RepeatedData(get_data('train',data_dir, meta_dir, batch_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            #aa = visualize_label(label)
            #pass
            cv2.imshow("im", im / 255.0)
            cv2.imshow("raw-label", label)
            cv2.imshow("color-label", visualize_label(label))
            cv2.waitKey(0)


def get_config(data_dir, meta_dir, batch_size):
    logger.auto_set_dir()
    dataset_train = get_data('train', data_dir, meta_dir, batch_size)
    steps_per_epoch = dataset_train.size() * 9
    dataset_val = get_data('val', data_dir, meta_dir, batch_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(2, 1.25e-4), (4, 5e-5), (6, 2.5e-5)]),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=1),
            ProgressBar(["cross_entropy_loss","cost","wd_cost"])#uncomment it to debug for every step
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=10,
    )


def run(model_path, image_path, output):
    return #TODO

class CalculateMIoU(Callback):
    def __init__(self, nb_class):
        self.nb_class = nb_class

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], ['prob'])

    def _before_train(self):
        pass

    def _trigger(self):
        global args
        self.val_ds = get_data('val', args.data_dir, args.meta_dir, args.batch_size)
        self.val_ds.reset_state()

        self.stat = MIoUStatistics(self.nb_class)

        for image, label in tqdm(self.val_ds.get_data()):
            label = np.squeeze(label)
            image = np.squeeze(image)
            prediction = predict_slider(image, self.pred, self.nb_class, tile_size=CROP_SIZE)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', default="/data1/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012",
                        help='dataset dir')
    parser.add_argument('--meta_dir', default="pascalvoc12", help='meta dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = 16, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data(args.data_dir,args.meta_dir,args.batch_size)
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config(args.data_dir,args.meta_dir,args.batch_size)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
