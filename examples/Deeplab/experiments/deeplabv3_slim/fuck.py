# Author: Tao Hu <taohu620@gmail.com>
from . import resnet_v2
import tensorflow as tf
slim = tf.contrib.slim

def fuck(inputs):
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_101(inputs,
                                              21,
                                              is_training=False,
                                              global_pool=False,
                                              output_stride=16,spatial_squeeze=False)
        return net