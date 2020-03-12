import tensorflow as tf

import resnet_utils
from resnet_v2 import bottleneck
import resnet_v2
slim = tf.contrib.slim

resnet_vn = 2

def deeplabv2(inputs,
              num_classes,
              aspp=True,
              reuse=None,
              is_training=True):
  scope ='resnet_v{}_101'.format(resnet_vn)
  with tf.variable_scope(scope, [inputs], reuse=reuse) as sc:
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
          net = inputs
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv2d],
                        activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')

          #18: ([2, 2, 2, 2], basicblock),
          #34: ([3, 4, 6, 3], basicblock),
          #50: ([3, 4, 6, 3], bottleneck),
          #101: ([3, 4, 23, 3], bottleneck),
          #152: ([3, 8, 36, 3], bottleneck)
          with tf.variable_scope('block1', [net]) as sc:
            base_depth = 64
            for i in range(2):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4,
                  depth_bottleneck=base_depth, stride=1)
            with tf.variable_scope('unit_3', values=[net]):
              net = bottleneck(net, depth=base_depth * 4,
                               depth_bottleneck=base_depth, stride=2)



          with tf.variable_scope('block2', [net]) as sc:
            base_depth = 128
            for i in range(3):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4,
                  depth_bottleneck=base_depth, stride=1)
            with tf.variable_scope('unit_4', values=[net]):
              net = bottleneck(net, depth=base_depth * 4,
                depth_bottleneck=base_depth, stride=2)



          with tf.variable_scope('block3', [net]) as sc:
            base_depth = 256
            for i in range(23):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4,
                  depth_bottleneck=base_depth,rate=2, stride=1)



          with tf.variable_scope('block4', [net]) as sc:
            base_depth = 512
            for i in range(3):
              with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                net = bottleneck(net, depth=base_depth * 4,
                  depth_bottleneck=base_depth, stride=1, rate=4)

          # This is needed because the pre-activation variant does not have batch
          # normalization or activation functions in the residual unit output. See
          # Appendix of [2].
          net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')




    with tf.variable_scope('logits',[net]) as sc:
      net = slim.conv2d(net, num_classes, [3,3], stride=1, rate=6,
        activation_fn=None, normalizer_fn=None)

    net = tf.image.resize_bilinear(net, inputs.shape[1:3])

    return net
