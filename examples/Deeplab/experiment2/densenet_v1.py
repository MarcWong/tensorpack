from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

import densenet_utils

slim = tf.contrib.slim
dense_arg_scope = densenet_utils.dense_arg_scope


def my_squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
  with tf.variable_scope(layer_name):
    squeeze = tf.reduce_mean(input_x, [1, 2], name='gap', keep_dims=False)

    with tf.variable_scope('fc1'):
      excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=int(out_dim / ratio))

    excitation = tf.nn.relu(excitation)

    with tf.variable_scope('fc2'):
      excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=out_dim)
    excitation = tf.nn.sigmoid(excitation)

    excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
    scale = input_x * excitation
    return scale

@slim.add_arg_scope
def unit(inputs, depth, kernel, stride=1, rate=1, drop=0):
  """Basic unit. BN -> RELU -> CONV
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The growth rate of the composite function layer.
           The num_outputs of bottleneck and transition layer.
    kernel: Kernel size.
    stride: The DenseNet unit's stride.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the DenseNet unit.
  Returns:
    The basic unit's output.
  """
  net = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
  net = slim.conv2d(net, num_outputs=depth, kernel_size=kernel, 
    stride=stride, rate=rate, scope='conv1')
  if drop > 0:
    net = slim.dropout(net, keep_prob=1-drop, scope='dropout')
  return net

@slim.add_arg_scope
def dense(inputs, growth, bottleneck=True, stride=1, rate=1, drop=0,
          outputs_collections=None, scope=None):
  """Dense layer.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    growth: The growth rate of the dense layer.
    bottleneck: Whether to use bottleneck.
    stride: The DenseNet unit's stride. Determines the amount of downsampling
    of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the dense layer.
    outputs_collections: Collection to add the dense layer output.
    scope: Optional variable_scope.
  Returns:
    The dense layer's output.
  """
  net = inputs
  if bottleneck:
    with tf.variable_scope('bottleneck', values=[net]):
      net = unit(net, depth=4*growth, kernel=[1,1], stride=stride, 
        rate=rate, drop=drop)
  
  with tf.variable_scope('composite', values=[net]):
    net = unit(net, depth=growth, kernel=[3,3], stride=stride, rate=rate, 
      drop=drop)

  return net

@slim.add_arg_scope
def transition(inputs,remove_latter_pooling, bottleneck=True, compress=0.5, stride=1, rate=1, drop=0,
               outputs_collections=None, scope=None):
  """Transition layer.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The transition layer's stride. Determines the amount of downsampling of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the transition layer.
    outputs_collections: Collection to add the transition layer output.
    scope: Optional variable_scope.
  Returns:
    The transition layer's output.
  """
  net = inputs

  if compress < 1:
    num_outputs = math.floor(inputs.get_shape().as_list()[3] * compress)
  else:
    num_outputs = inputs.get_shape().as_list()[3]

  net = unit(net, depth=num_outputs, kernel=[1,1], stride=1,
        rate=rate)
  if stride > 1:
      net = slim.avg_pool2d(net, kernel_size=[2,2], stride=stride, scope='avg_pool')
  else:
    if not remove_latter_pooling:
      net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=stride, scope='avg_pool')

  if drop > 0:
    net = slim.dropout(net, keep_prob=1-drop, scope='dropout')

  return net

@slim.add_arg_scope
def stack_dense_blocks(inputs, blocks, growth, remove_latter_pooling, senet, denseindense, bottleneck=True, compress=0.5,
  stride=1, rate=1, drop=0, outputs_collections=None, scope=None):
  """Dense block.
  Args:
    inputs: A tensor of size [batch, height, width, channels].
    blocks: List of number of layers in each block.
    growth: The growth rate of the dense layer.
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
    rate: An integer, rate for atrous convolution.
    drop: The dropout rate of the transition layer.
    outputs_collections: Collection to add the dense layer output.
    scope: Optional variable_scope.
  Returns:
    The dense block's output.
  """
  net = inputs
  denseindense_list = []
  for i, num_layer in enumerate(blocks):
    with tf.variable_scope('block%d' %(i+1), [net]) as sc_block:
      for j in range(num_layer):
        with tf.variable_scope('dense%d' %(j+1), values=[net]) as sc_layer:
          identity = tf.identity(net)
          dense_output= dense(net, growth, bottleneck, stride = 1, rate= rate[i], drop = 0) # disable dropout in dense conv;

          if senet > 0:
            output_dim = dense_output.get_shape().as_list()[-1]
            dense_output = my_squeeze_excitation_layer(dense_output,output_dim,ratio=senet,layer_name='seblock{}'.format(j+1))#TODO SENet

          net = tf.concat([identity, dense_output], axis=3,
            name='concat%d' %(j+1))

      net = slim.utils.collect_named_outputs(outputs_collections, 
        sc_block.name, net)

    if i < len(blocks) - 1: # last block doesn't have transition
      denseindense_list.append(net)
      with tf.variable_scope('trans%d' %(i+1), values=[net]) as sc_trans:
        net = transition(net,remove_latter_pooling, bottleneck, compress, stride[i], rate=1, drop=0)# enable dropout in transition;
        net = slim.utils.collect_named_outputs(outputs_collections, 
          sc_trans.name, net)

  if denseindense:
    with tf.variable_scope('denseindense'):
      to_be_concated = []
      for ii,did in enumerate(denseindense_list):
        did = tf.image.resize_bilinear(did, net.shape[1:3])
        did = slim.batch_norm(did, activation_fn=tf.nn.relu, scope='preact{}'.format(ii+1))
        did = slim.conv2d(did, num_outputs=did.get_shape().as_list()[-1]//8, kernel_size=3,
                          stride=1, rate=1, scope='conv{}'.format(ii+1))
        did = my_squeeze_excitation_layer(did,did.get_shape().as_list()[-1],4,"se{}".format(ii+1))
        to_be_concated.append(did)
      net = tf.concat(to_be_concated, axis=3,
                      name='concat')
      return net
  else:
    return net

def densenet(inputs,
             blocks,
             rate,
             stride,
             weight_decay,
             growth=32,
             bottleneck=True,
             compress=0.5,
             drop=0,
             stem = 0,
             senet = 0,
             denseindense = 0,
             num_classes=None,
             is_training=True,
             data_name=None,
             remove_latter_pooling = False,
             reuse=None,
             scope=None):
  """Generator for DenseNet models.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of DenseNet blocks. Each 
    element is a densenet_utils.DenseBlock object describing the units in the 
    block.
    growth: The growth rate of the DenseNet unit.
    bottleneck: Whether to use bottleneck.
    compress: The compression ratio of the transition layer.
    stride: The dense layer's stride. Determines the amount of downsampling of the units output compared to its input.
    drop: The dropout rate of the transition layer.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: Whether batch_norm and drop_out layers are in training mode.
    data_name: Which type of model to use.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If num_classes is 0 or None, then net is the output of the last DenseNet
      block, potentially after global average pooling. If num_classes is a 
      non-zero integer, net contains the pre-softmax activations.
    end_points: A dictionary from components of the network to the 
    corresponding activation.
  """

  assert len(stride) == len(blocks)
  assert len(rate) == len(blocks)

  with tf.variable_scope(scope, 'densenet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(dense_arg_scope(weight_decay=weight_decay)):
      with slim.arg_scope([slim.conv2d, slim.batch_norm, stack_dense_blocks],
                          outputs_collections=end_points_collection):
        with slim.arg_scope([slim.batch_norm, slim.dropout], 
          is_training=is_training):
          net = inputs
            
          if data_name is 'imagenet':
            if stem == 0 :
              net = slim.conv2d(net, growth * 2, kernel_size=[7, 7], stride=2,
                                scope='conv1')
              net = slim.max_pool2d(net, [3, 3], padding='SAME', stride=2,
                                    scope='pool1')
            elif stem == 1:
              net = slim.conv2d(net, 64, kernel_size=[3, 3], stride=2,
                                scope='conv1')
              net = slim.conv2d(net, 64, kernel_size=[3, 3], stride=1,
                                scope='conv1_1')
              net = slim.conv2d(net, 128, kernel_size=[3, 3], stride=1,
                                scope='conv1_2')
              net = slim.max_pool2d(net, [2, 2], padding='SAME', stride=2,
                                    scope='pool1')
            else:
              raise
          else:
            net = slim.conv2d(net, growth*2, kernel_size=[3, 3], stride=2, 
              scope='conv1')
          
          net = stack_dense_blocks(net, blocks, growth, remove_latter_pooling, senet,denseindense, bottleneck, compress,
            stride, rate, drop)

          net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
          # Convert end_points_collection into a dictionary of end_points.
          end_points = slim.utils.convert_collection_to_dict(
              end_points_collection)

          net = slim.conv2d(net, num_outputs=num_classes, kernel_size=1,
                            stride=1, rate=6, scope='conv2classnum') # dilation 2,4,6
          net = tf.image.resize_bilinear(net, inputs.shape[1:3]) #upsample 4x

          return net, end_points

def densenet_121(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_121'], 
    data_name='imagenet')

def densenet_169(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_169'],
    data_name='imagenet')

def densenet_201(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_201'],
    data_name='imagenet')

def densenet_265(inputs):
  return densenet(inputs, blocks=densenet_utils.networks['densenet_265'],
    data_name='imagenet')


if __name__ == "__main__":
  x = tf.placeholder(tf.float32, [None, 224, 224, 3])

  net, end_points = densenet_121(x)

  for i in end_points:
    print(end_points[i])
