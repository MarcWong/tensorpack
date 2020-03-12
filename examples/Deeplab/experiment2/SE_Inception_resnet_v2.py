# https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_Inception_resnet_v2.py

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

import numpy as np

weight_decay = 0.0005
momentum = 0.9
init_learning_rate = 0.1

reduction_ratio = 4
batch_size = 128
iteration = 391
# 128 * 391 ~ 50,000

test_iteration = 10

total_epochs = 100

def conv_layer(input, filter, kernel, stride=1, padding='SAME', layer_name="conv", activation=True):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=True, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        if activation :
            network = Relu(network)
        return network

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Max_pooling(x, pool_size=[3,3], stride=2, padding='VALID') :
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Dropout(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)


def squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = Global_Average_Pooling(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name + '_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name + '_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input_x * excitation
        return scale

