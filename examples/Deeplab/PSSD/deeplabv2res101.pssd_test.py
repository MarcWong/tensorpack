#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import cv2
import tensorflow as tf
import argparse
from six.moves import zip
import os
import numpy as np

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.segmentation.segmentation import imwrite_grid, visualize_label, predict_scaler
from tensorpack.utils.stats import MIoUStatistics
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tqdm import tqdm
from seg_utils import RandomCropWithPadding, softmax_cross_entropy_with_ignore_label
from resnet_model import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck_deeplab, se_resnet_bottleneck,
    resnet_backbone)


CLASS_NUM = dataset.PSSD.class_num()
CROP_SIZE = 513
IGNORE_LABEL = 255

first_batch_lr = 2.5e-4
lr_schedule = [(3, 1e-4), (7, 1e-5)]
epoch_scale = 30
max_epoch = 1
lr_multi_schedule = [('aspp.*_conv/W', 5),('aspp.*_conv/b',10)]
batch_size = 1
evaluate_every_n_epoch = 1

TEST_DIR="/data1/Dataset/heda/shizhai"

class Model(ModelDesc):

    def __init__(self, height=CROP_SIZE, width=CROP_SIZE):
        super(Model, self).__init__()
        self.height = height
        self.width = width

    def _get_inputs(self):
        ## Set static shape so that tensorflow knows shape at compile time.
        return [InputDesc(tf.float32, [None, self.height, self.width, 3], 'image'),
                InputDesc(tf.int32, [None, self.height, self.width], 'gt')]

    def _build_graph(self, inputs):
        def resnet101(image):
            mode = 'resnet'
            depth = 101
            basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
            bottleneck = {
                'resnet': resnet_bottleneck_deeplab,
                'preact': preresnet_bottleneck,
                'se': se_resnet_bottleneck}[mode]
            num_blocks, block_func = {
                18: ([2, 2, 2, 2], basicblock),
                34: ([3, 4, 6, 3], basicblock),
                50: ([3, 4, 6, 3], bottleneck),
                101: ([3, 4, 23, 3], bottleneck),
                152: ([3, 8, 36, 3], bottleneck)
            }[depth]

            def get_logits(image):
                with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format="NHWC"):
                    return resnet_backbone(
                        image, num_blocks,
                        preresnet_group if mode == 'preact' else resnet_group, block_func,CLASS_NUM,ASPP = False)

            return get_logits(image)

        image, label = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        label = tf.identity(label, name="label")

        predict = resnet101(image)

        costs = []
        prob = tf.nn.softmax(predict, name='prob')

        label4d = tf.expand_dims(label, 3, name='label4d')
        new_size = prob.get_shape()[1:3]


        cost = softmax_cross_entropy_with_ignore_label(logits=predict, label=label4d,
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
            #add_moving_summary(costs + [self.cost])

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=first_batch_lr, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=2.5e-4)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                lr_multi_schedule)])


def get_data(name, base_dir, meta_dir, batch_size):
    isTrain = True if 'train' in name else False
    ds = dataset.PSSD(base_dir, meta_dir, name, shuffle=True)


    if isTrain:#special augmentation
        shape_aug = [imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                            aspect_ratio_thres=0.15),
                     RandomCropWithPadding(CROP_SIZE,IGNORE_LABEL),
                     imgaug.Flip(horiz=True),
                     ]
    else:
        shape_aug = []

    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)


    if isTrain:
        ds = BatchData(ds, batch_size)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data(base_dir,meta_dir, batch_size):
    ds = RepeatedData(get_data('train',base_dir, meta_dir, batch_size), -1)
    ds.reset_state()
    for ims, labels in ds.get_data():
        for im, label in zip(ims, labels):
            #aa = visualize_label(label)
            #pass
            cv2.imshow("im", im / 255.0)
            cv2.imshow("raw-label", label)
            cv2.imshow("color-label", visualize_label(label))
            cv2.waitKey(0)


def get_config( base_dir, meta_dir, batch_size):
    logger.auto_set_dir()
    nr_tower = max(get_nr_gpu(), 1)

    dataset_train = get_data('train', base_dir, meta_dir, batch_size)
    steps_per_epoch = dataset_train.size() * epoch_scale


    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=evaluate_every_n_epoch),
            ProgressBar(["cross_entropy_loss","cost","wd_cost"])#uncomment it to debug for every step
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch,
    )


def run(model_path, image_path, output):
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['output' + str(k) for k in range(1, 7)])
    predictor = OfflinePredictor(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    im = cv2.resize(
        im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16)
    )[None, :, :, :].astype('float32')
    outputs = predictor(im)
    if output is None:
        for k in range(6):
            pred = outputs[k][0]
            cv2.imwrite("out{}.png".format(
                '-fused' if k == 5 else str(k + 1)), pred * 255)
    else:
        pred = outputs[5][0]
        cv2.imwrite(output, pred * 255)

def proceed_test_dir(args):
    import cv2

    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['prob'])
    predictor = OfflinePredictor(pred_config)

    from tensorpack.utils.fs import mkdir_p
    src_dir = os.path.join(TEST_DIR,"images")
    visual_dir = os.path.join(TEST_DIR,"visualization")
    final_dir = os.path.join(TEST_DIR,"final")
    predict_dir = os.path.join(TEST_DIR,"prediction")
    import shutil
    shutil.rmtree(visual_dir, ignore_errors=True)
    shutil.rmtree(final_dir, ignore_errors=True)
    shutil.rmtree(predict_dir, ignore_errors=True)

    mkdir_p(visual_dir)
    mkdir_p(final_dir)
    mkdir_p(predict_dir)

    ll = os.listdir(src_dir)

    logger.info("start forwarding....")

    def mypredictor(input_img):
        # input image: 1*H*W*3
        # output : H*W*C
        output = predictor(input_img[np.newaxis, :, :, :])
        return output[0][0]

    for i in tqdm(range(len(ll))):
        filename = ll[i]
        image = cv2.imread(os.path.join(src_dir,filename))

	prediction = predict_scaler(image, mypredictor, scales=[0.5,0.75, 1, 1.25, 1.5], classes=CLASS_NUM, tile_size=CROP_SIZE, is_densecrf = False)
        np.save(os.path.join(predict_dir,"{}".format(filename.replace(".JPG",".npy"))), prediction)
        prediction = np.argmax(prediction, axis=2)
        cv2.imwrite(os.path.join(final_dir,"{}".format(filename)), prediction)
        cv2.imwrite(os.path.join(visual_dir, "{}".format(filename)), np.concatenate((image, visualize_label(prediction)), axis=1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="1", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--base_dir', default="/data1/Dataset/UDD", help='base dir')
    parser.add_argument('--meta_dir', default="../metadata/UDD", help='meta dir')
    parser.add_argument('--load', default="train_log/deeplabv2res101.pssd_train/model-27000", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = batch_size, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    parser.add_argument('--test', action='store_true', help='generate test result')
    parser.add_argument('--test_dir', default=TEST_DIR, help='generate test result')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data(args.base_dir, args.meta_dir,args.batch_size)
    elif args.run:
        run(args.load, args.run, args.output)
    elif args.test_dir:
        proceed_test_dir(args)
