#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf

from datetime import datetime


class Trainer(object):
    def __init__(self, model, data_loader, log_path='./logs'):
        self.model = model
        self.data_loader = data_loader
        self.full_log_path = os.path.abspath(log_path)
        if not os.path.exists(self.full_log_path):
            os.mkdir(self.full_log_path)

    def train(self, params, initialize_model=True):

        tf.set_random_seed(params['random_seed'])

        features = tf.placeholder(
            tf.float32,
            [
                params['batch_size'],
                *self.data_loader.get_img_shape()
            ]
        )
        labels = tf.placeholder(
            tf.int64, params['batch_size']
        )

        logits, others = self.model(features)

        with tf.name_scope('loss'):
            ind_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits
            )
            loss = tf.reduce_mean(ind_loss) + params['l2'] * others['l2_norm']
            tf.summary.scalar('loss', loss)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope('optimizer'):
            with tf.control_dependencies(update_ops):
                step = tf.train.AdamOptimizer(params['lr']).minimize(loss)

        with tf.name_scope('error'):
            correct = tf.equal(tf.argmax(logits, 1), labels)
            correct = tf.cast(correct, tf.float32)
            error = 1.0 - tf.reduce_mean(correct)
            tf.summary.scalar('error', error)

        train_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'log_%s/train' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
            )
        )
        validate_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'log_%s/validation' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
            )
        )
        # train_writer.add_graph(tf.get_default_graph())
        validate_writer.add_graph(tf.get_default_graph())

        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            if initialize_model:
                sess.run(tf.global_variables_initializer())

            cumulative_train_batch_idx = 0
            for epoch_idx in range(params['n_epoch']):
                print(
                    '[%s] Epoch %d - Parameters: %s' %
                    (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        epoch_idx,
                        str(params)
                    )
                )

                generators = self.data_loader.reset_batch_generators(params['batch_size'])
                train_batch_generator, validate_batch_generator, test_batch_generator = generators
                print(
                    '[%s] Epoch %d - Training Summary: ' %
                    (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        epoch_idx
                    ),
                    end=""
                )
                cumulative_train_batch_idx, _, _ = self.epoch(
                    cumulative_train_batch_idx,
                    [features, labels, others['keep_prob'], others['is_training']],
                    [step, loss, error, merged],
                    sess,
                    train_batch_generator,
                    train_writer,
                    params
                )
                print(
                    '[%s] Epoch %d - Validation Summary: ' %
                    (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        epoch_idx
                    ),
                    end=""
                )
                self.epoch(
                    cumulative_train_batch_idx,
                    [features, labels, others['keep_prob'], others['is_training']],
                    [loss, error, merged],
                    sess,
                    validate_batch_generator,
                    validate_writer,
                    params
                )

    @staticmethod
    def epoch(acc_batch_idx, placeholders, run_list, session, generator, log_writer, params):
        features, labels, keep_prob, is_training = placeholders
        loss_list = list()
        error_list = list()
        for batch_idx, batch_features, batch_labels in generator:
            result = session.run(
                run_list,
                feed_dict={
                    features: batch_features,
                    labels: batch_labels,
                    keep_prob: params['keep_prob'],
                    is_training: any(['optimizer' in repr(item) for item in run_list])
                }
            )
            loss_list.append(result[-3])
            error_list.append(result[-2])
            log_writer.add_summary(result[-1], acc_batch_idx)
            acc_batch_idx += 1
        print(
            'loss: %f \t error rate: %s' %
            (
                float(np.mean(loss_list)),
                float(np.mean(error_list))
            )
        )
        return acc_batch_idx, np.mean(loss_list), np.mean(error_list)
