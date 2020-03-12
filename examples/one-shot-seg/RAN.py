# Author: Tao Hu <taohu620@gmail.com>

# -*- coding: utf-8 -*-
"""
attention module of Residual Attention Network
"""

import tensorflow as tf
from resnet_v2 import bottleneck


def my_residualblock(out,input_channels,scope=None):
    return bottleneck(out, depth=input_channels, depth_bottleneck=input_channels // 4, stride=1,
               scope=scope)

def AttentionModule(input, input_channels, scope="attention_module"):
        p = 1
        t = 2
        r = 1

        """
        f_prop function of attention module
        :param input: A Tensor. input data [batch_size, height, width, channel]
        :param input_channels: dimension of input channel.
        :param scope: str, tensorflow name scope
        :param is_training: boolean, whether training step or not(test step)
        :return: A Tensor [batch_size, height, width, channel]
        """
        with tf.variable_scope(scope):

            # residual blocks(TODO: change this function)
            with tf.variable_scope("first_residual_blocks"):
                for i in range(p):
                    input = my_residualblock(input, input_channels, scope="num_blocks_{}".format(i))

            with tf.variable_scope("trunk_branch"):
                output_trunk = input
                for i in range(t):
                    output_trunk = my_residualblock(output_trunk, input_channels, scope="num_blocks_{}".format(i))

            with tf.variable_scope("soft_mask_branch"):

                with tf.variable_scope("down_sampling_1"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(input, ksize=filter_, strides=filter_, padding='SAME')

                    for i in range(r):
                        output_soft_mask = my_residualblock(output_soft_mask, input_channels, scope="num_blocks_{}".format(i))

                with tf.variable_scope("skip_connection"):
                    # TODO(define new blocks)
                    output_skip_connection = my_residualblock(output_soft_mask, input_channels)


                with tf.variable_scope("down_sampling_2"):
                    # max pooling
                    filter_ = [1, 2, 2, 1]
                    output_soft_mask = tf.nn.max_pool(output_soft_mask, ksize=filter_, strides=filter_, padding='SAME')

                    for i in range(r):
                        output_soft_mask = my_residualblock(output_soft_mask, input_channels, scope="num_blocks_{}".format(i))

                # smallest feature map

                with tf.variable_scope("up_sampling_1"):
                    for i in range(r):
                        output_soft_mask = my_residualblock(output_soft_mask, input_channels, scope="num_blocks_{}".format(i))

                    # interpolation
                    output_soft_mask = tf.image.resize_bilinear(output_soft_mask, (output_soft_mask.shape[1]*2,output_soft_mask.shape[2]*2),name="upsample")

                # add skip connection
                output_soft_mask += output_skip_connection

                with tf.variable_scope("up_sampling_2"):
                    for i in range(r):
                        output_soft_mask = my_residualblock(output_soft_mask, input_channels, scope="num_blocks_{}".format(i))

                    # interpolation
                    output_soft_mask = tf.image.resize_bilinear(output_soft_mask, (
                    output_soft_mask.shape[1] * 2, output_soft_mask.shape[2] * 2), name="upsample")


                with tf.variable_scope("output"):
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)
                    output_soft_mask = tf.layers.conv2d(output_soft_mask, filters=input_channels, kernel_size=1)

                    # sigmoid
                    output_soft_mask = tf.nn.sigmoid(output_soft_mask)

            with tf.variable_scope("attention"):
                output = (1 + output_soft_mask) * output_trunk

            with tf.variable_scope("last_residual_blocks"):
                for i in range(p):
                    output = my_residualblock(output, input_channels, scope="num_blocks_{}".format(i))

            return output
