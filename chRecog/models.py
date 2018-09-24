#!/usr/bin/env python

import numpy as np
import tensorflow as tf


def head(x, is_training):
    with tf.name_scope('preprocessing'):
        x = tf.cast(x, tf.float32)
        x = tf.layers.batch_normalization(
            x,
            moving_mean_initializer=tf.constant_initializer(234.0),
            moving_variance_initializer=tf.constant_initializer(3300.0),
            training=is_training
        )
    return x


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    if tf.executing_eagerly():
        return initial
    else:
        return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    if tf.executing_eagerly():
        return initial
    else:
        return tf.Variable(initial)


def conv2d(indata, w, padding='SAME'):
    return tf.nn.conv2d(
        indata,
        w,
        strides=[1, 1, 1, 1],
        padding=padding
    )


def max_pooling(indata, psize=3, stride=3, padding='SAME'):
    return tf.nn.max_pool(
        indata,
        ksize=[1, psize, psize, 1],
        strides=[1, stride, stride, 1],
        padding=padding
    )


def visualize(idx, kernel):
    with tf.variable_scope('visualization'):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        tf.summary.image('layer_%d' % idx, kernel_transposed, max_outputs=64)


def conv_part(idx, indata, height_k_size, width_k_size, in_size, out_size, show_visualization,
              p_stride=3, padding='SAME', keep_prob=None, l2_norm_collector=None):
    with tf.name_scope('conv_%d' % idx):
        if keep_prob is not None:
            with tf.name_scope('dropout'):
                indata = tf.nn.dropout(indata, keep_prob)
        w_conv = weight_variable([height_k_size, width_k_size, in_size, out_size])
        b_conv = bias_variable([out_size])
        h_conv = tf.nn.relu(conv2d(indata, w_conv, padding=padding) + b_conv)
        if show_visualization:
            visualize(idx, w_conv)
    with tf.name_scope('pool'):
        h_pool = max_pooling(h_conv, stride=p_stride, padding=padding)
    if l2_norm_collector is not None:
        l2_norm_collector.append(
            tf.reduce_mean(
                tf.square(w_conv)
            )
        )
    return h_pool


def conv_fc_part(idx, indata, height_k_size, width_k_size, in_size, out_size, activation, keep_prob=None):
    with tf.name_scope('conv_fc_%d' % idx):
        if keep_prob is not None:
            with tf.name_scope('dropout'):
                indata = tf.nn.dropout(indata, keep_prob)
        w_fc = weight_variable([height_k_size, width_k_size, in_size, out_size])
        b_fc = bias_variable([out_size])
        if activation == 'relu':
            h_fc = tf.nn.relu(conv2d(indata, w_fc, padding='VALID') + b_fc)
        elif activation == 'sigmoid':
            h_fc = tf.nn.sigmoid(conv2d(indata, w_fc, padding='VALID') + b_fc)
        else:
            h_fc = conv2d(indata, w_fc, padding='VALID') + b_fc
    return h_fc


def fcn(node, is_training=None, keep_prob=None):
    if is_training is None:
        is_training = False
    if keep_prob is None:
        keep_prob = 0.85

    node = head(node, is_training)

    l2_norm_list = list()

    if tf.executing_eagerly():
        node = conv_part(
            1, node, height_k_size=12, width_k_size=8, in_size=3, out_size=64, p_stride=3,
            show_visualization=False, l2_norm_collector=l2_norm_list, keep_prob=None
        )
    else:
        node = conv_part(
            1, node, height_k_size=12, width_k_size=8, in_size=3, out_size=64, p_stride=3,
            show_visualization=True, l2_norm_collector=l2_norm_list, keep_prob=None
        )

    node = conv_part(
        2, node, height_k_size=8, width_k_size=6, in_size=64, out_size=96, p_stride=2,
        show_visualization=False, l2_norm_collector=l2_norm_list, keep_prob=keep_prob
    )

    node = conv_part(
        3, node, height_k_size=6, width_k_size=4, in_size=96, out_size=128, p_stride=2,
        show_visualization=False, l2_norm_collector=l2_norm_list, keep_prob=keep_prob
    )

    node = conv_fc_part(
        1, node, height_k_size=6, width_k_size=3, in_size=128, out_size=1024,
        activation='relu', keep_prob=keep_prob
    )

    node = conv_fc_part(
        1, node, height_k_size=1, width_k_size=1, in_size=1024, out_size=512,
        activation='relu', keep_prob=keep_prob
    )

    node = conv_fc_part(
        2, node, height_k_size=1, width_k_size=1, in_size=512, out_size=90,
        activation='no', keep_prob=None
    )

    total_l2_norm = tf.add_n(l2_norm_list) / float(len(l2_norm_list))

    return node, {'l2_norm': total_l2_norm}


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.enable_eager_execution()
    # data = tf.random_normal([4, 360, 360, 3])
    data = tf.random_normal([4, 64, 32, 3])
    node = data
    o, _ = fcn(data)
