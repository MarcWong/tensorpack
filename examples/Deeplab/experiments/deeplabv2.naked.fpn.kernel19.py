#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: deeplabv2.py
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
from tensorpack.utils.segmentation.segmentation import visualize_uncertainty, visualize_label, predict_scaler
from tensorpack.utils.stats import MIoUStatistics,MIoUBoundaryStatistics
from tensorpack.dataflow.imgaug.misc import RandomCropWithPadding
from tensorpack.utils import logger
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
import tensorpack.tfutils.symbolic_functions as symbf
from tqdm import tqdm


from resnet_model_fpn import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck,
    resnet_group, resnet_basicblock, resnet_bottleneck, se_resnet_bottleneck,
    resnet_backbone)


CLASS_NUM = dataset.PascalVOC12.class_num()
CROP_SIZE = 512
IGNORE_LABEL = 255

class Model(ModelDesc):

    def _get_inputs(self):
        ## Set static shape so that tensorflow knows shape at compile time.
        return [InputDesc(tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'image'),
                InputDesc(tf.int32, [None, CROP_SIZE, CROP_SIZE], 'gt')]

    def _build_graph(self, inputs):
        def resnet101(image, label):
            mode = 'resnet'
            depth = 101
            basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
            bottleneck = {
                'resnet': resnet_bottleneck,
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
                        preresnet_group if mode == 'preact' else resnet_group, block_func,CLASS_NUM)

            return get_logits(image)

        image, label = inputs
        image = image - tf.constant([104, 116, 122], dtype='float32')
        label = tf.identity(label, name="label")

        predict = resnet101(image, label)

        costs = []
        prob = tf.nn.softmax(predict, name='prob')

        label4d = tf.expand_dims(label, 3, name='label4d')
        new_size = prob.get_shape()[1:3]
        #label_resized = tf.image.resize_nearest_neighbor(label4d, new_size)

        cost = symbf.softmax_cross_entropy_with_ignore_label(logits=predict, label=label4d,
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
        opt = tf.train.AdamOptimizer(lr, epsilon=2.5e-4)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('fpn.*W', 10),('fpn.*b',20)])])


def get_data(name, data_dir, meta_dir, batch_size):
    isTrain = name == 'train'
    ds = dataset.PascalVOC12(data_dir, meta_dir, name, shuffle=True)

    if isTrain:
        shape_aug = [
           imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                                             aspect_ratio_thres=0.15),
            RandomCropWithPadding(CROP_SIZE,IGNORE_LABEL),
            imgaug.Flip(horiz=True),
        ]
    else:
        shape_aug = []
        pass
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(ds):
        return ds
    if isTrain:
        ds = MapData(ds, f)
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
    steps_per_epoch = dataset_train.size() * 8
    dataset_val = get_data('val', data_dir, meta_dir, batch_size)

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(2, 1e-4), (4, 1e-5), (6, 8e-6)]),
            HumanHyperParamSetter('learning_rate'),
            PeriodicTrigger(CalculateMIoU(CLASS_NUM), every_k_epochs=1),
            ProgressBar(["cross_entropy_loss","cost","wd_cost"])#uncomment it to debug for every step
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=10,
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

def proceed_validation(args, is_save = True, is_densecrf = False):
    import cv2
    ds = dataset.PascalVOC12(args.data_dir, args.meta_dir, "val")
    ds = BatchData(ds, 1)

    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(args.load),
        input_names=['image'],
        output_names=['prob'])
    predictor = OfflinePredictor(pred_config)

    i = 0
    from tensorpack.utils.fs import mkdir_p
    result_dir = os.path.join("baseline_result_on_{}".format("val"))
    mkdir_p(result_dir)

    #stat = MIoUStatistics(CLASS_NUM)
    stat = MIoUBoundaryStatistics(CLASS_NUM,kernel=19)
    logger.info("start validation....")
    for image, label in tqdm(ds.get_data()):
        label = np.squeeze(label)
        image = np.squeeze(image)
        prediction_prob = predict_scaler(image, predictor, scales=[0.9, 1, 1.1], classes=CLASS_NUM, tile_size=CROP_SIZE, is_densecrf = is_densecrf)
        prediction = np.argmax(prediction_prob, axis=2)
        stat.feed(prediction, label)

        if is_save:
            cv2.imwrite(os.path.join(result_dir, "{}.png".format(i)),image)
            cv2.imwrite(os.path.join(result_dir, "{}-label.png".format(i)), label)
            cv2.imwrite(os.path.join(result_dir, "{}-predict.png".format(i)), visualize_label(prediction))
            cv2.imwrite(os.path.join(result_dir, "{}-uncertainty.png".format(i)), visualize_uncertainty(prediction_prob,label))

            #cv2.imwrite(os.path.join(result_dir,"{}.png".format(i)), np.concatenate((image, visualize_label(label), visualize_label(prediction)), axis=1))

        i += 1
    stat.print_result()





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
            prediction = predict_scaler(image, self.pred, scales=[0.9, 1, 1.1], classes=CLASS_NUM, tile_size=CROP_SIZE,
                           is_densecrf=False)
            prediction = np.argmax(prediction, axis=2)
            self.stat.feed(prediction, label)

        self.trainer.monitors.put_scalar("mIoU", self.stat.mIoU)
        self.trainer.monitors.put_scalar("mean_accuracy", self.stat.mean_accuracy)
        self.trainer.monitors.put_scalar("accuracy", self.stat.accuracy)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="4", help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data_dir', default="/data2/dataset/pascalvoc2012/VOC2012trainval/VOCdevkit/VOC2012",
                        help='dataset dir')
    parser.add_argument('--meta_dir', default="../metadata/pascalvoc12", help='meta dir')
    parser.add_argument('--load', default="../resnet101.npz", help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--batch_size', type=int, default = 10, help='batch_size')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    parser.add_argument('--validation', action='store_true', help='validate model on validation images')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    if args.view:
        view_data(args.data_dir,args.meta_dir,args.batch_size)
    elif args.run:
        run(args.load, args.run, args.output)
    elif args.validation:
        proceed_validation(args)
    else:
        config = get_config(args.data_dir,args.meta_dir,args.batch_size)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_nr_gpu(), 1)))
