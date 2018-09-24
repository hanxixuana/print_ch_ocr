#!/usr/bin/env python

import os
import json
import inspect
import numpy as np
import tensorflow as tf

from datetime import datetime


class Trainer(object):
    def __init__(self, data_loader, log_path='./logs'):
        self.data_loader = data_loader
        self.full_log_path = os.path.join(
            os.path.abspath(log_path),
            'log_%s' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        if not os.path.exists(os.path.abspath(log_path)):
            os.mkdir(os.path.abspath(log_path))

    def train(self, model, params, initialize_model=True):

        starting_string = (
                '[%s] Started for parameters: %s -> %s' %
                (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    str(params),
                    self.full_log_path
                )
        )
        print(starting_string)

        tf.set_random_seed(params['random_seed'])

        features = tf.placeholder(
            tf.float32,
            [
                None, None, None, 3
            ],
            name='features'
        )
        labels = tf.placeholder(
            tf.int64,
            [None],
            name='labels'
        )
        is_training = tf.placeholder(tf.bool, name='is_training')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        logits, others = model(features, is_training, keep_prob)

        logits = tf.identity(logits, name='logits')

        if len(logits.shape) > 2:
            with tf.name_scope('logits_reshape'):
                logits = tf.reshape(
                    logits, [tf.shape(logits)[0], tf.reduce_prod(tf.shape(logits)[1:])]
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
            lr = tf.placeholder(tf.float32, name='lr')
            with tf.control_dependencies(update_ops):
                if 'optimizer' in params:
                    if params['optimizer'] == 'gd':
                        step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
                    else:
                        step = tf.train.AdamOptimizer(lr).minimize(loss)
                else:
                    step = tf.train.AdamOptimizer(lr).minimize(loss)

        with tf.name_scope('error'):
            correct = tf.equal(tf.argmax(logits, 1), labels)
            correct = tf.cast(correct, tf.float32)
            error = 1.0 - tf.reduce_mean(correct)
            tf.summary.scalar('error', error)

            t3correct = tf.nn.in_top_k(logits, labels, 3)
            t3correct = tf.cast(t3correct, tf.float32)
            t3_error = 1.0 - tf.reduce_mean(t3correct)
            tf.summary.scalar('top-3-error', t3_error)

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
        if not os.path.exists(os.path.join(self.full_log_path, 'saved_models')):
            os.mkdir(os.path.join(self.full_log_path, 'saved_models'))
        with open(os.path.join(self.full_log_path, 'params.json'), 'w') as file:
            json.dump(params, file)
        with open(os.path.join(self.full_log_path, 'model_settings.txt'), 'w') as file:
            file.write(inspect.getsource(model))
        with open(os.path.join(self.full_log_path, 'record.txt'), 'w') as file:
            file.write(starting_string + '\n')

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

                cumulative_batch_idx, train_loss, train_error_rate, train_t3_error_rate = self.epoch(
                    cumulative_batch_idx,
                    epoch_idx,
                    [features, labels, keep_prob, is_training, lr],
                    [step, loss, error, t3_error, merged],
                    sess,
                    train_batch_generator,
                    train_writer,
                    params
                )
                _, validate_loss, validate_error_rate, validate_t3_error_rate = self.epoch(
                    cumulative_batch_idx,
                    epoch_idx,
                    [features, labels, keep_prob, is_training, lr],
                    [loss, error, t3_error, merged],
                    sess,
                    validate_batch_generator,
                    validate_writer,
                    params
                )

                for_recording = (
                        '[%s] Epoch %d - Parameters: %s -> %s\n'
                        'train loss: %f train error: %f train top-3 error: %f '
                        'validate loss: %f validate error: %f validate top-3 error: %f' %
                        (
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                            epoch_idx,
                            str(params),
                            self.full_log_path,
                            train_loss,
                            train_error_rate,
                            train_t3_error_rate,
                            validate_loss,
                            validate_error_rate,
                            validate_t3_error_rate
                        )
                )
                print(for_recording)
                with open(os.path.join(self.full_log_path, 'record.txt'), 'a') as file:
                    file.write(for_recording + '\n')

                if epoch_idx % 5 == 0:
                    tf.saved_model.simple_save(
                        sess,
                        os.path.join(
                            self.full_log_path,
                            'saved_models',
                            'epoch_%d' % epoch_idx
                        ),
                        inputs={
                            'features': features,
                            'labels': labels,
                            'keep_prob': keep_prob,
                            'is_training': is_training
                        },
                        outputs={
                            'logits': logits,
                            'l2_norm': others['l2_norm']
                        }
                    )

    @staticmethod
    def epoch(cum_batch_idx, epoch_idx, placeholders, run_list, session, generator, log_writer, params, verbose=False):
        features, labels, keep_prob, is_training, lr = placeholders

        learning_rate = params['lr'][
            max(
                [key for key in params['lr'].keys() if key <= epoch_idx]
            )
        ]

        loss_list = list()
        error_list = list()
        t3_error_list = list()
        for _, batch_features, batch_labels in generator:
            result = session.run(
                run_list,
                feed_dict={
                    features: batch_features,
                    labels: batch_labels,
                    keep_prob: params['keep_prob'],
                    is_training: any(['optimizer' in repr(item) for item in run_list]),
                    lr: learning_rate
                }
            )
            loss_list.append(result[-4])
            error_list.append(result[-3])
            t3_error_list.append(result[-2])
            log_writer.add_summary(result[-1], cum_batch_idx)
            cum_batch_idx += 1
        if verbose:
            print(
                'loss: %f \t error rate: %s \t top-3 error rate: %s' %
                (
                    float(np.mean(loss_list)),
                    float(np.mean(error_list)),
                    float(np.mean(t3_error_list))
                )
            )
        return cum_batch_idx, float(np.mean(loss_list)), float(np.mean(error_list)), float(np.mean(t3_error_list))
