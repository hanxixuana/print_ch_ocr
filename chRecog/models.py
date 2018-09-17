#!/usr/bin/env python

import numpy as np
import tensorflow as tf


def deepnn(x):

    def conv2d(x, w):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_3x3(x):
        """max_pool_3x3 downsamples a feature map by 3X."""
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                              strides=[1, 3, 3, 1], padding='SAME')

    def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    x = tf.cast(x, tf.float32)

    is_training = tf.placeholder(tf.bool)
    x = tf.layers.batch_normalization(
        x,
        moving_mean_initializer=tf.constant_initializer(234.0),
        moving_variance_initializer=tf.constant_initializer(3300.0),
        training=is_training
    )

    # x = x - tf.constant(234.0)
    # x = x / tf.constant(57.0)

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([8, 8, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)

        # tf.summary.histogram("w_conv1", w_conv1)
        # tf.summary.histogram("b_conv1", b_conv1)
        # tf.summary.histogram("h_conv1", h_conv1)
        # visualize(w_conv1)

    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool1'):
        h_pool1 = max_pool_3x3(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([8, 8, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

        # tf.summary.histogram("w_conv2", w_conv2)
        # tf.summary.histogram("b_conv2", b_conv2)
        # tf.summary.histogram("h_conv2", h_conv2)

    # Second pooling layer.
    with tf.name_scope('pool2'):
        h_pool2 = max_pool_3x3(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([np.prod(h_pool2.shape[1:]).value, 2056])
        b_fc1 = bias_variable([2056])

        h_pool2_flat = tf.reshape(h_pool2, [-1, np.prod(h_pool2.shape[1:])])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([2056, 90])
        b_fc2 = bias_variable([90])

        y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

    l2_norm = (
        tf.reduce_mean(w_conv1 ** 2.0)
        +
        tf.reduce_mean(w_conv2 ** 2.0)
    )

    return y_conv, {'l2_norm': l2_norm, 'keep_prob': keep_prob, 'is_training': is_training}


def visualize(kernel):
    with tf.variable_scope('visualization'):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        tf.summary.image('filters', kernel_transposed, max_outputs=16)
