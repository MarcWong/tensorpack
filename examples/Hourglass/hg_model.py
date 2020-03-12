# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from resnet_v1 import bottleneck_hourglass
from resnet_utils import conv2d_same, resnet_arg_scope



def create_bottleneck(prefix, f_in, stride, depth_bottleneck, depth, has_proj=False):
        with tf.variable_scope(prefix, [f_in]) as sc:
            return bottleneck_hourglass(f_in,depth,depth_bottleneck,stride)


def add_init(module, module_name):
    input_name = 'input'
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None, normalizer_fn=None):
        f = conv2d_same(module[input_name], 64, 7, stride=1, scope=module_name + '_1_convbnrelu')
    f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name = module_name + '_1_pool')
    f = create_bottleneck(module_name + '_bottleneck1', f, 1, 64, 128, has_proj=True)

    f = tf.nn.max_pool(f, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME', name=module_name + '_2')
    f = create_bottleneck(module_name + '_bottleneck2', f, 1, 64, 128, has_proj=False)
    f = create_bottleneck(module_name + '_bottleneck3', f, 1, 128, 256, has_proj=True)

    module[module_name] = f


def add_hourglass(module, module_name, n, num_channels, input_name) :  #n_max = 4, f_max = 256
    this_name = '{}{}'.format(module_name, n)
    middle_channels = int(num_channels / 2)
    up1 = create_bottleneck('{}_up1'.format(this_name), module[input_name], 1, middle_channels, num_channels, has_proj=False)

    f = tf.nn.max_pool(module[input_name], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    module[this_name+'_low1'] = create_bottleneck('{}_low1'.format(this_name), f, 1, middle_channels, num_channels, has_proj=False)

    if n > 1 :
        add_hourglass(module, module_name, n-1, num_channels, '{}_low1'.format(this_name))
        low2 = module['{}{}'.format(module_name,str(n-1))]
    else :# n = 1
        module['{}_low2'.format(this_name)] = create_bottleneck('{}_low2'.format(this_name), module[this_name+'_low1'], 1, middle_channels, num_channels, has_proj=False) # f changed to module[this_name+'_low1'] by dongzhuoyao
        low2 = module['{}_low2'.format(this_name)]


    low3 = create_bottleneck('{}_low3'.format(this_name), low2, 1, middle_channels, num_channels, has_proj=False)
    up2 = tf.image.resize_images(low3, [tf.cast(low3.shape[1],tf.int32)*2,tf.cast(low3.shape[2],tf.int32)*2])
    module[this_name] = up1 + up2
    if n == 4 :
        module[module_name] = module[this_name]
    return module[this_name]

def make_network(data, stage, nr_skeleton, is_training):
    with slim.arg_scope(resnet_arg_scope(weight_decay=1e-4)):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                module = {}
                output = []
                module['input'] = data
                add_init(module, 'init_1')
                last_input = 'init_1'
                for i in range(stage) :
                    f = add_hourglass(module, 'hg{}_scale'.format(i), 4, 256, last_input)
                    f = create_bottleneck('hg{}_1'.format(i), f, 1, 128, 256, has_proj=False)
                    f = slim.conv2d(f, 256, [1, 1], stride=1,scope='hg{}_2_convbnrelu'.format(i)) #cbr
                    module['hg{}_2'.format(i)] = f
                    module['output_{}'.format(i)] = slim.conv2d(f, nr_skeleton, [1, 1], stride=1,
                                                                scope='output_{}'.format(i), activation_fn=None,
                                                                normalizer_fn=None)
                    output.append(module['output_'+str(i)])

                    if i < stage - 1 :
                        module['hg{}_3'.format(i)] = slim.conv2d(module['hg{}_2'.format(i)], 256, [1, 1],
                                                                 stride=1,scope='hg{}_3'.format(i),activation_fn=None, normalizer_fn=None)
                        module['hg{}_4'.format(i)] = slim.conv2d(module['output_{}'.format(i)], 256, [1, 1], stride=1,
                                                                    scope='hg{}_4'.format(i), activation_fn=None,
                                                                    normalizer_fn=None)

                        module['hg{}'.format(i)] = module['hg{}_3'.format(i)] + module['hg{}_4'.format(i)]
                        module['hg{}'.format(i)] = module['hg{}'.format(i)] + module[last_input]
                        last_input = 'hg{}'.format(i)

    return output





