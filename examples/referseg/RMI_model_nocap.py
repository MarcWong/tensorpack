import numpy as np
import tensorflow as tf
import sys

from deeplab_resnet import model as deeplab101

from util.processing_tools import *



class RMI_model(object):

    def __init__(self, img, class_num, batch_size,
                        num_steps, mode, vocab_size,
                        vf_h = 40,
                        vf_w = 40,
                        H = 320,
                        W = 320,
                        vf_dim = 2048,
                        w_emb_dim = 1000,
                        v_emb_dim = 1000,
                        mlp_dim = 500,
                        rnn_size = 1000,
                        keep_prob_rnn = 1.0,
                        keep_prob_emb = 1.0,
                        keep_prob_mlp = 1.0,
                        num_rnn_layers = 1,
                        weights = 'resnet'):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.mlp_dim = mlp_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.mode = mode
        self.weights = weights
        self.class_num = class_num

        self.im = img #tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])

        if self.weights == 'resnet':
            resmodel = resnet_model.ResNet(batch_size=self.batch_size, 
                                        atrous=True,
                                        images=self.im,
                                        labels=tf.constant(0.))
            self.visual_feat = resmodel.logits

        elif self.weights == 'deeplab':
            resmodel = deeplab101.DeepLabResNetModel({'data': self.im}, is_training=False)
            self.visual_feat = resmodel.layers['res5c_relu']

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            #self.train_op()

    def build_graph(self):

        if self.weights == 'deeplab':
            # atrous0 = self._atrous_conv("atrous0", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 6)
            # atrous1 = self._atrous_conv("atrous1", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 12)
            # atrous2 = self._atrous_conv("atrous2", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 18)
            # atrous3 = self._atrous_conv("atrous3", self.visual_feat, 3, self.vf_dim, self.v_emb_dim, 24)
            # visual_feat = tf.add(atrous0, atrous1)
            # visual_feat = tf.add(visual_feat, atrous2)
            # visual_feat = tf.add(visual_feat, atrous3)
            visual_feat = self._conv("conv0", self.visual_feat, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
        elif self.weights == 'resnet':
            visual_feat = self.visual_feat

        mlp = tf.nn.relu(visual_feat)
        conv2 = self._conv("conv2", mlp, 1, mlp.get_shape().as_list()[-1], self.class_num, [1, 1, 1, 1])
        self.pred = conv2
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], 
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    """
    def train_op(self):
        # define loss
        self.target = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg') or var.op.name.startswith('ResNet/fc1000')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0]
        self.cls_loss = loss.weighed_logistic_loss(self.pred, self.target)
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss + self.reg_loss

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001, power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {var: (2.0 if var.op.name.find(r'biases') > 0 else 1.0) for var in tvars}
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)
    """