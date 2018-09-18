#!/usr/bin/env python

import os
import json
import numpy as np
import tensorflow as tf

from datetime import datetime


class Trainer(object):
    def __init__(self, model, data_loader, log_path='./logs'):
        self.model = model
        self.data_loader = data_loader
        self.full_log_path = os.path.join(
            os.path.abspath(log_path),
            'log_%s' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        if not os.path.exists(os.path.abspath(log_path)):
            os.mkdir(os.path.abspath(log_path))

    def train(self, params, initialize_model=True):

        print(
            '[%s] Started for parameters: %s -> %s' %
            (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                str(params),
                self.full_log_path
            )
        )

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

        if len(logits.shape) > 2:
            with tf.name_scope('logits_reshape'):
                logits = tf.reshape(
                    logits, [-1, np.prod(logits.shape[1:])]
                )

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
                if 'optimizer' in params:
                    if params['optimizer'] == 'gd':
                        step = tf.train.GradientDescentOptimizer(params['lr']).minimize(loss)
                    else:
                        step = tf.train.AdamOptimizer(params['lr']).minimize(loss)
                else:
                    step = tf.train.AdamOptimizer(params['lr']).minimize(loss)

        with tf.name_scope('error'):
            correct = tf.equal(tf.argmax(logits, 1), labels)
            correct = tf.cast(correct, tf.float32)
            error = 1.0 - tf.reduce_mean(correct)
            tf.summary.scalar('error', error)

        train_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'train'
            )
        )
        validate_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'validate'
            )
        )
        validate_writer.add_graph(tf.get_default_graph())
        with open(os.path.join(self.full_log_path, 'params.json'), 'w') as file:
            json.dump(params, file)
        with open(os.path.join(self.full_log_path, 'model_settings.json'), 'w') as file:
            json.dump([str(item) for item in tf.trainable_variables()], file)

        merged = tf.summary.merge_all()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            if initialize_model:
                sess.run(tf.global_variables_initializer())

            cumulative_batch_idx = 0
            for epoch_idx in range(params['n_epoch']):
                generators = self.data_loader.reset_batch_generators(params['batch_size'])
                train_batch_generator, validate_batch_generator, test_batch_generator = generators

                cumulative_batch_idx, train_loss, train_error_rate = self.epoch(
                    cumulative_batch_idx,
                    [features, labels, others['keep_prob'], others['is_training']],
                    [step, loss, error, merged],
                    sess,
                    train_batch_generator,
                    train_writer,
                    params
                )
                _, validate_loss, validate_error_rate = self.epoch(
                    cumulative_batch_idx,
                    [features, labels, others['keep_prob'], others['is_training']],
                    [loss, error, merged],
                    sess,
                    validate_batch_generator,
                    validate_writer,
                    params
                )

                print(
                    '[%s] Epoch %d - Parameters: %s -> %s\n'
                    'train loss: %f\t train error: %f\t '
                    'validate loss: %f\t validate error: %f' %
                    (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        epoch_idx,
                        str(params),
                        self.full_log_path,
                        train_loss,
                        train_error_rate,
                        validate_loss,
                        validate_error_rate
                    )
                )

    @staticmethod
    def epoch(cum_batch_idx, placeholders, run_list, session, generator, log_writer, params, verbose=False):
        features, labels, keep_prob, is_training = placeholders
        loss_list = list()
        error_list = list()
        for _, batch_features, batch_labels in generator:
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
            log_writer.add_summary(result[-1], cum_batch_idx)
            cum_batch_idx += 1
        if verbose:
            print(
                'loss: %f \t error rate: %s' %
                (
                    float(np.mean(loss_list)),
                    float(np.mean(error_list))
                )
            )
        return cum_batch_idx, float(np.mean(loss_list)), float(np.mean(error_list))
