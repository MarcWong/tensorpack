#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

import numpy as np
from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap, AtrousConv2D, MaxPooling)


def resnet_shortcut(l, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, stride=stride, nl=nl)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name: BatchNorm('bn', x, gamma_init=tf.zeros_initializer())
    else:
        return lambda x, name: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, nl=get_bn(zero_init=False))


def resnet_bottleneck_deeplab(l, ch_out, stride, dilation, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    if dilation == 1:
        l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    else:
        l = AtrousConv2D('conv2', l, ch_out, kernel_shape=3, rate=dilation, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))

def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride, dilation, stride_first):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1, dilation, stride_first)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l

def edge_conv(l, name, channel_num):
    channel_axis = 3
    W_init = tf.contrib.layers.variance_scaling_initializer()
    W1 = tf.get_variable('{}_W1'.format(name), [3,3,channel_num,1], initializer=W_init)
    constant1 = np.array([[1,1,1],[0,0,0],[1,1,1]])
    constant1 = np.expand_dims(constant1, axis=-1)
    constant1 = np.expand_dims(constant1, axis=-1)
    constant1 = np.broadcast_to(constant1,(3,3,channel_num,1))
    W1_mask = tf.constant(constant1,dtype=tf.float32)
    W1 = W1*W1_mask

    W2 = tf.get_variable('{}_W2'.format(name), [3,3,channel_num,1], initializer=W_init)
    constant2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    constant2 = np.expand_dims(constant2, axis=-1)
    constant2 = np.expand_dims(constant2, axis=-1)
    constant2 = np.broadcast_to(constant2, (3, 3, channel_num, 1))
    W2_mask = tf.constant(constant2, dtype=tf.float32)

    W2 = W2 * W2_mask


    inputs = tf.split(l, channel_num, channel_axis)


    kernels1 = tf.split(W1, channel_num, 2)
    output = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding="SAME")
                   for i, k in zip(inputs, kernels1)]
    conv1 = tf.concat(output, channel_axis)

    kernels2 = tf.split(W2, channel_num, 2)
    output = [tf.nn.conv2d(i, k, strides=[1, 1, 1, 1], padding="SAME")
              for i, k in zip(inputs, kernels2)]
    conv2 = tf.concat(output, channel_axis)

    l = conv1*conv1 + conv2*conv2 + l
    return l

def resnet_backbone(image, num_blocks, group_func, block_func,class_num, ASPP = False):
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=variance_scaling_initializer(mode='FAN_OUT')):
        l = Conv2D('conv0', image,  64, 7, stride=2, nl=BNReLU)
        l = MaxPooling('pool0', l,  shape=3, stride=2, padding='SAME')
        l = group_func(l, 'group0', block_func, 64, num_blocks[0], 1, dilation=1, stride_first=False)
        l = edge_conv(l,name="edge_conv0",channel_num=256)
        l = group_func(l, 'group1', block_func, 128, num_blocks[1], 2, dilation=1, stride_first=True)
        l = edge_conv(l, name="edge_conv1",channel_num=512)
        l = group_func(l, 'group2', block_func, 256, num_blocks[2], 2, dilation=2, stride_first=True)
        l = edge_conv(l, name="edge_conv2",channel_num=1024)
        resnet_head = group_func(l, 'group3', block_func, 512, num_blocks[3], 1, dilation=4, stride_first=False)
        resnet_head = edge_conv(resnet_head, name="edge_conv3", channel_num=2048)

    def aspp_branch(input, rate):
        input = AtrousConv2D('aspp{}_conv'.format(rate), input, class_num, kernel_shape=3, rate=rate)
        return input
    if ASPP:
        output = aspp_branch(resnet_head , 6) +aspp_branch(resnet_head, 12) +aspp_branch(resnet_head, 18)+aspp_branch(resnet_head, 24)
    else:
        output = aspp_branch(resnet_head, 6)
    output = tf.image.resize_bilinear(output, image.shape[1:3])
    return output
