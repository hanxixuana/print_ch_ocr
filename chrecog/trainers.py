#!/usr/bin/env python

import os
import json
import inspect
import numpy as np
import tensorflow as tf

from copy import deepcopy
from datetime import datetime

from abc import ABCMeta, abstractmethod

from chrecog.ctcmodels import DecoderType


class Trainer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def train(self, model_or_path, params):
        raise NotImplementedError


class CTCNetTrainer(Trainer):

    saved_model_name = 'model'

    def __init__(self, data_loader, log_path='./logs'):

        super(CTCNetTrainer, self).__init__()

        self.data_loader = data_loader

        self.full_log_path = os.path.join(
            os.path.abspath(log_path),
            'log_%s' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        if not os.path.exists(os.path.abspath(log_path)):
            os.mkdir(os.path.abspath(log_path))

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.params = None

        self.graph = None
        self.sess = None
        self.saver = None

        self.path_to_pretrained = None
        self.model = None

        self.features = None
        self.label_tensor_cell_idx = None
        self.label_tensor_char_idx = None
        self.label_tensor_dense_shape = None
        self.seq_len = None
        self.is_training = None
        self.keep_prob = None
        self.logits = None
        self.l2_norm = None

        self.loss = None
        self.lr = None
        self.step = None

        self.decoder = None

        self.merged = None

    @staticmethod
    def check_params(params):
        param_names = [
            'n_epoch', 'max_text_len', 'n_epoch', 'batch_size',
            'lr', 'l2', 'random_seed', 'img_height', 'img_width'
        ]
        missing_param_names = list()
        for name in param_names:
            if name not in params:
                missing_param_names.append(name)
        if len(missing_param_names) > 0:
            raise ValueError(
                'params should contain %s.' %
                ', '.join(missing_param_names)
            )

    @staticmethod
    def to_sparse(char_list, texts):
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    @staticmethod
    def decoder_output_to_text(char_list, ctc_output):

        batch_size = ctc_output[1].shape[0]

        # contains string of labels for each batch element
        encoded_label_strs = [[] for _ in range(batch_size)]

        decoded = ctc_output[0][0]

        # go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]  # index according to [b,t]
            encoded_label_strs[batch_element].append(label)

        result = [
            str().join(
                [
                    char_list[c]
                    for c in labelStr
                ]
            )
            for labelStr in encoded_label_strs
        ]

        return result

    @staticmethod
    def setup_decoder(ctc_in_3d, seq_len, decoder_type):
        # BxTxC -> TxBxC
        ctc_in_3d_tbc = tf.transpose(ctc_in_3d, [1, 0, 2])

        # decoder: either best path decoding or beam search decoding
        if decoder_type == DecoderType.BestPath:
            decoder = tf.nn.ctc_greedy_decoder(
                inputs=ctc_in_3d_tbc,
                sequence_length=seq_len
            )
        elif decoder_type == DecoderType.BeamSearch:
            decoder = tf.nn.ctc_beam_search_decoder(
                inputs=ctc_in_3d_tbc,
                sequence_length=seq_len,
                beam_width=50,
                merge_repeated=False
            )
        else:
            raise ValueError(
                'Unknown decoder type: %s.' %
                decoder_type
            )

        return decoder

    def train(self, model_or_path, params=None):
        """
        Launch the trainer with a path to a trained model or a new model.

        trained model:  model_or_path is the path to the folder of four files
                        1. checkpoint
                        2. '%s.data-00000-of-00001' % self.saved_model_name
                        3. '%s.index' % self.saved_model_name
                        4. '%s.meta' % self.saved_model_name

        new model:  model_or_path is a function satisfying:
                    f(x, is_training, keep_prob) -> y, l2_norm

        :param callable or str model_or_path:       Model to be trained
        :param dict params:                         Parameters for training
        """
        if not isinstance(params, dict):
            raise ValueError(
                'params should be a dictionary of parameters.'
            )
        self.check_params(params)
        self.params = deepcopy(params)

        if isinstance(model_or_path, str):
            # TODO: not used or tested
            if not os.path.exists(model_or_path):
                raise FileNotFoundError(
                    'Cannot find the path %s.' %
                    model_or_path
                )
            if '%s.meta' % self.saved_model_name not in os.listdir(model_or_path):
                raise FileNotFoundError(
                    'Cannot find %s.meta in the folder pointed to by %s' %
                    (
                        self.saved_model_name,
                        model_or_path
                    )
                )
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph, config=self.config)
            self.saver = tf.train.import_meta_graph(
                os.path.join(
                    model_or_path,
                    '%s.meta' % self.saved_model_name
                )
            )
            self.saver.restore(
                self.sess,
                tf.train.latest_checkpoint(model_or_path)
            )
            self.path_to_pretrained = model_or_path
            self.model = None

            self.features = self.graph.get_tensor_by_name('features:0')
            self.label_tensor_cell_idx = self.graph.get_tensor_by_name('label_tensor_cell_idx:0')
            self.label_tensor_char_idx = self.graph.get_tensor_by_name('label_tensor_char_idx:0')
            self.label_tensor_dense_shape = self.graph.get_tensor_by_name('label_tensor_dense_shape:0')
            self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
            self.is_training = self.graph.get_tensor_by_name('is_training:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.logits = self.graph.get_tensor_by_name('logits:0')
            self.l2_norm = self.graph.get_tensor_by_name('l2_norm:0')

            if len(self.logits.shape) > 2:
                with self.graph.name_scope('logits_reshape'):
                    self.logits = tf.reshape(
                        self.logits,
                        [
                            tf.shape(self.logits)[0],
                            tf.reduce_prod(
                                tf.shape(self.logits)[1:]
                            )
                        ]
                    )

            self.loss = self.graph.get_tensor_by_name('loss:0')
            self.lr = self.graph.get_tensor_by_name('lr:0')
            if 'optimizer' in params:
                if params['optimizer'] == 'adam':
                    self.step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                elif params['optimizer'] == 'rmsprop':
                    self.step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                else:
                    raise ValueError(
                        'Specify a wrong optimizer: %s. Choose one from adam or rmsprop.' %
                        params['optimizer']
                    )
            else:
                self.step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.decoder = self.setup_decoder(
                self.logits, self.seq_len, params['decoder_type']
            )

            self.merged = self.graph.get_tensor_by_name('merged:0')

        elif callable(model_or_path):

            self.graph = tf.Graph()

            with self.graph.as_default() as g:
                tf.set_random_seed(params['random_seed'])

                self.path_to_pretrained = None
                self.model = model_or_path

                self.features = tf.placeholder(
                    tf.float32,
                    [
                        None, params['img_height'], params['img_width'], 3
                    ],
                    name='features'
                )
                self.label_tensor_cell_idx = tf.placeholder(
                    tf.int64,
                    shape=[None, 2],
                    name='label_tensor_cell_idx'
                )
                self.label_tensor_char_idx = tf.placeholder(
                    tf.int32,
                    [None],
                    name='label_tensor_char_idx'
                )
                self.label_tensor_dense_shape = tf.placeholder(
                    tf.int64,
                    [2],
                    name='label_tensor_dense_shape'
                )
                labels = tf.SparseTensor(
                    self.label_tensor_cell_idx,
                    self.label_tensor_char_idx,
                    self.label_tensor_dense_shape,
                )
                self.seq_len = tf.placeholder(
                    tf.int32,
                    [None],
                    name='seq_len'
                )
                self.is_training = tf.placeholder(
                    tf.bool,
                    name='is_training'
                )
                self.keep_prob = tf.placeholder(
                    tf.float32,
                    name='keep_prob'
                )

                self.logits, self.l2_norm = model_or_path(
                    self.features,
                    self.is_training,
                    self.keep_prob
                )
                self.logits = tf.identity(
                    self.logits,
                    name='logits'
                )
                self.l2_norm = tf.identity(
                    self.l2_norm,
                    name='l2_norm'
                )

                with g.name_scope('loss_part'):
                    # calc loss for batch
                    self.loss = tf.add(
                        tf.reduce_mean(
                            tf.nn.ctc_loss(
                                labels=labels,
                                inputs=self.logits,
                                sequence_length=self.seq_len,
                                ctc_merge_repeated=True,
                                time_major=False,
                            )
                        ),
                        tf.multiply(
                            params['l2'],
                            self.l2_norm
                        )
                    )
                self.loss = tf.identity(self.loss, name='loss')

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.lr = tf.placeholder(tf.float32, name='lr')
                with tf.control_dependencies(update_ops):
                    if 'optimizer' in params:
                        if params['optimizer'] == 'adam':
                            self.step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
                        elif params['optimizer'] == 'rmsprop':
                            self.step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
                        else:
                            raise ValueError(
                                'Specify a wrong optimizer: %s. Choose one from adam or rmsprop.' %
                                params['optimizer']
                            )
                    else:
                        self.step = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

                with g.name_scope('decoder'):
                    self.decoder = self.setup_decoder(
                        self.logits, self.seq_len, params['decoder_type']
                    )

                with g.name_scope('measures'):
                    tf.summary.scalar('loss', self.loss)
                self.merged = tf.identity(tf.summary.merge_all(), name='merged')

                self.saver = tf.train.Saver()

            self.sess = tf.Session(
                graph=self.graph,
                config=self.config
            )

        else:
            raise ValueError(
                'model should be either a path to a saved model '
                'or a function that defines a model.'
            )

        self.__train()

    def __train(self):

        ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        starting_string = (
                '[%s] <--- %s \n%s ---> %s' %
                (
                    ct,
                    str(self.params),
                    ' ' * (len(ct) + 2),
                    self.full_log_path
                )
        )
        print(starting_string)

        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'train'
            )
        )
        train_writer.add_graph(tf.get_default_graph())
        validate_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'validate'
            )
        )

        if not os.path.exists(os.path.join(self.full_log_path, 'saved_models')):
            os.mkdir(os.path.join(self.full_log_path, 'saved_models'))
        with open(os.path.join(self.full_log_path, 'params.json'), 'w') as file:
            json.dump(self.params, file)
        with open(os.path.join(self.full_log_path, 'record.txt'), 'w') as file:
            file.write(starting_string + '\n')
        if self.model is not None:
            with open(os.path.join(self.full_log_path, 'model_settings.txt'), 'w') as file:
                file.write(inspect.getsource(self.model))
        else:
            with open(os.path.join(self.full_log_path, 'model_settings.txt'), 'w') as file:
                file.write(str(self.graph.as_graph_def()))

        cumulative_batch_idx = 0
        for epoch_idx in range(self.params['n_epoch']):
            generators = self.data_loader.reset_batch_generators(self.params['batch_size'])
            train_batch_generator, validate_batch_generator, test_batch_generator = generators

            cumulative_batch_idx, train_loss, train_error_rate = self.epoch(
                cumulative_batch_idx,
                epoch_idx,
                [
                    self.features,
                    self.label_tensor_cell_idx,
                    self.label_tensor_char_idx,
                    self.label_tensor_dense_shape,
                    self.seq_len,
                    self.is_training,
                    self.keep_prob,
                    self.lr
                ],
                [
                    self.step,
                    self.decoder,
                    self.loss,
                    self.merged
                ],
                self.sess,
                train_batch_generator,
                train_writer,
                self.params,
                verbose=self.params['verbose']
            )

            _, validate_loss, validate_error_rate = self.epoch(
                cumulative_batch_idx,
                epoch_idx,
                [
                    self.features,
                    self.label_tensor_cell_idx,
                    self.label_tensor_char_idx,
                    self.label_tensor_dense_shape,
                    self.seq_len,
                    self.is_training,
                    self.keep_prob,
                    self.lr
                ],
                [
                    self.decoder,
                    self.loss,
                    self.merged
                ],
                self.sess,
                validate_batch_generator,
                validate_writer,
                self.params
            )

            ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            for_recording = (
                    '[%s] Epoch %d <--- %s \n%s ---> %s \n%s ---> '
                    'train loss: %f train error: %f '
                    'validate loss: %f validate error: %f' %
                    (
                        ct,
                        epoch_idx,
                        str(self.params),
                        ' ' * (len(ct) + 2),
                        self.full_log_path,
                        ' ' * (len(ct) + 2),
                        train_loss,
                        train_error_rate,
                        validate_loss,
                        validate_error_rate,
                    )
            )
            print(for_recording)
            with open(os.path.join(self.full_log_path, 'record.txt'), 'a') as file:
                file.write(for_recording + '\n')

            if epoch_idx % 5 == 0:
                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.full_log_path,
                        'saved_models',
                        'epoch_%d' % epoch_idx,
                        self.saved_model_name
                    )
                )

            if np.isnan(train_loss):
                print(
                    'Got nan loss. Quit.'
                )
                return

    def epoch(self, cum_batch_idx, epoch_idx, placeholders, run_list,
              session, generator, log_writer, params, verbose=False):
        (
            features, label_tensor_cell_idx, label_tensor_char_idx,
            label_tensor_dense_shape, seq_len, is_training, keep_prob, lr
        ) = placeholders

        learning_rate = params['lr'][
            max(
                [key for key in params['lr'].keys() if key <= epoch_idx]
            )
        ]

        loss_list = list()
        error_list = list()
        for _, batch_features, batch_labels in generator:
            cell_idx, char_idx, shape = self.to_sparse(params['char_list'], batch_labels)
            result = session.run(
                run_list,
                feed_dict={
                    features: batch_features,
                    label_tensor_cell_idx: cell_idx,
                    label_tensor_char_idx: char_idx,
                    label_tensor_dense_shape: shape,
                    seq_len: [params['max_text_len']] * params['batch_size'],
                    is_training: any(['optimizer' in repr(item) for item in run_list]),
                    keep_prob: params['keep_prob'],
                    lr: learning_rate
                }
            )
            loss_list.append(result[-2])
            cum_batch_idx += 1
            if cum_batch_idx % ((epoch_idx + 1) * 10) == 0:
                log_writer.add_summary(result[-1], cum_batch_idx)
            decoded = self.decoder_output_to_text(params['char_list'], result[-3])
            correct = 0
            for predict, target in zip(batch_labels, decoded):
                if predict == target:
                    correct += 1
            error = 1.0 - correct / params['batch_size']
            error_list.append(
                error
            )
            # if verbose and epoch_idx == 0:
            if verbose:
                ct = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print(
                    '[%s] ---> %d loss: %f error: %f' %
                    (
                        ct,
                        cum_batch_idx,
                        float(result[-2]),
                        float(error),
                    )
                )
        return cum_batch_idx, float(np.mean(loss_list)), float(np.mean(error_list))


class ConvNetTrainer(Trainer):

    saved_model_name = 'model'

    def __init__(self, data_loader, log_path='./logs'):

        super(ConvNetTrainer, self).__init__()

        self.data_loader = data_loader

        self.full_log_path = os.path.join(
            os.path.abspath(log_path),
            'log_%s' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        if not os.path.exists(os.path.abspath(log_path)):
            os.mkdir(os.path.abspath(log_path))

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True

        self.params = None

        self.graph = None
        self.sess = None
        self.saver = None

        self.path_to_pretrained = None
        self.model = None

        self.features = None
        self.labels = None
        self.keep_prob = None
        self.is_training = None
        self.logits = None
        self.l2_norm = None

        self.loss = None
        self.lr = None
        self.step = None
        self.error = None
        self.t3_error = None

        self.merged = None

    @staticmethod
    def check_params(params):
        param_names = [
            'n_epoch', 'batch_size', 'lr', 'l2', 'keep_prob', 'random_seed', 'optimizer'
        ]
        missing_param_names = list()
        for name in param_names:
            if name not in params:
                missing_param_names.append(name)
        if len(missing_param_names) > 0:
            raise ValueError(
                'params should contain %s.' %
                ', '.join(missing_param_names)
            )

    def train(self, model_or_path, params=None):
        """
        Launch the trainer with a path to a trained model or a new model.

        trained model:  model_or_path is the path to the folder of four files
                        1. checkpoint
                        2. '%s.data-00000-of-00001' % self.saved_model_name
                        3. '%s.index' % self.saved_model_name
                        4. '%s.meta' % self.saved_model_name

        new model:  model_or_path is a function satisfying:
                    f(x, is_training, keep_prob) -> y, l2_norm

        :param callable or str model_or_path:       Model to be trained
        :param dict params:                         Parameters for training
        """
        if not isinstance(params, dict):
            raise ValueError(
                'params should be a dictionary of parameters.'
            )
        self.check_params(params)
        self.params = deepcopy(params)
        if isinstance(model_or_path, str):
            # TODO: not used or tested
            if not os.path.exists(model_or_path):
                raise FileNotFoundError(
                    'Cannot find the path %s.' %
                    model_or_path
                )
            if '%s.meta' % self.saved_model_name not in os.listdir(model_or_path):
                raise FileNotFoundError(
                    'Cannot find %s.meta in the folder pointed to by %s' %
                    (
                        self.saved_model_name,
                        model_or_path
                    )
                )
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph, config=self.config)
            self.saver = tf.train.import_meta_graph(
                os.path.join(
                    model_or_path,
                    '%s.meta' % self.saved_model_name
                )
            )
            self.saver.restore(
                self.sess,
                tf.train.latest_checkpoint(model_or_path)
            )
            self.path_to_pretrained = model_or_path
            self.model = None

            self.features = self.graph.get_tensor_by_name('features:0')
            self.labels = self.graph.get_tensor_by_name('labels:0')
            self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
            self.is_training = self.graph.get_tensor_by_name('is_training:0')
            self.logits = self.graph.get_tensor_by_name('logits:0')
            self.l2_norm = self.graph.get_tensor_by_name('l2_norm:0')

            if len(self.logits.shape) > 2:
                with self.graph.name_scope('logits_reshape'):
                    self.logits = tf.reshape(
                        self.logits,
                        [
                            tf.shape(self.logits)[0],
                            tf.reduce_prod(
                                tf.shape(self.logits)[1:]
                            )
                        ]
                    )

            self.loss = self.graph.get_tensor_by_name('loss:0')
            self.lr = self.graph.get_tensor_by_name('lr:0')
            self.step = self.graph.get_operation_by_name('step')
            self.error = self.graph.get_tensor_by_name('error:0')
            self.t3_error = self.graph.get_tensor_by_name('t3_error:0')

            self.merged = self.graph.get_tensor_by_name('merged:0')

        elif callable(model_or_path):

            self.graph = tf.Graph()

            with self.graph.as_default() as g:

                self.path_to_pretrained = None
                self.model = model_or_path

                self.features = tf.placeholder(
                    tf.float32,
                    [
                        None, None, None, 3
                    ],
                    name='features'
                )
                self.labels = tf.placeholder(
                    tf.int64,
                    [None],
                    name='labels'
                )
                self.is_training = tf.placeholder(
                    tf.bool,
                    name='is_training'
                )
                self.keep_prob = tf.placeholder(
                    tf.float32,
                    name='keep_prob'
                )

                self.logits, self.l2_norm = model_or_path(
                    self.features,
                    self.is_training,
                    self.keep_prob
                )
                self.logits = tf.identity(
                    self.logits,
                    name='logits'
                )
                self.l2_norm = tf.identity(
                    self.l2_norm,
                    name='l2_norm'
                )

                if len(self.logits.shape) > 2:
                    with g.name_scope('logits_reshape'):
                        self.logits = tf.reshape(
                            self.logits,
                            [
                                tf.shape(self.logits)[0],
                                tf.reduce_prod(
                                    tf.shape(self.logits)[1:]
                                )
                            ]
                        )

                with g.name_scope('loss_part'):
                    ind_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.labels,
                        logits=self.logits
                    )
                    self.loss = (
                            tf.reduce_mean(ind_loss)
                            +
                            self.params['l2']
                            *
                            self.l2_norm
                    )
                self.loss = tf.identity(self.loss, name='loss')

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.lr = tf.placeholder(tf.float32, name='lr')
                with tf.control_dependencies(update_ops):
                    if 'optimizer' in self.params:
                        if self.params['optimizer'] == 'gd':
                            self.step = tf.train.GradientDescentOptimizer(self.lr, name='step').minimize(self.loss)
                        else:
                            self.step = tf.train.AdamOptimizer(self.lr, name='step').minimize(self.loss)
                    else:
                        self.step = tf.train.AdamOptimizer(self.lr, name='step').minimize(self.loss)

                with g.name_scope('error_part'):
                    correct = tf.equal(tf.argmax(self.logits, 1), self.labels)
                    correct = tf.cast(correct, tf.float32)
                    self.error = 1.0 - tf.reduce_mean(correct)

                    t3correct = tf.nn.in_top_k(self.logits, self.labels, 3)
                    t3correct = tf.cast(t3correct, tf.float32)
                    self.t3_error = 1.0 - tf.reduce_mean(t3correct)

                self.error = tf.identity(self.error, name='error')
                self.t3_error = tf.identity(self.t3_error, name='t3_error')

                with g.name_scope('measures'):
                    tf.summary.scalar('loss', self.loss)
                    tf.summary.scalar('error', self.error)
                    tf.summary.scalar('top-3-error', self.t3_error)
                self.merged = tf.identity(tf.summary.merge_all(), name='merged')

                self.saver = tf.train.Saver()

            self.sess = tf.Session(
                graph=self.graph,
                config=self.config
            )

            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())

        else:
            raise ValueError(
                'model should be either a path to a saved model '
                'or a function that defines a model.'
            )

        self.__train()

    def __train(self):

        starting_string = (
                '[%s] Started for parameters: %s -> %s' %
                (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    str(self.params),
                    self.full_log_path
                )
        )
        print(starting_string)

        tf.set_random_seed(self.params['random_seed'])

        train_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'train'
            )
        )
        train_writer.add_graph(tf.get_default_graph())
        validate_writer = tf.summary.FileWriter(
            os.path.join(
                self.full_log_path,
                'validate'
            )
        )

        if not os.path.exists(os.path.join(self.full_log_path, 'saved_models')):
            os.mkdir(os.path.join(self.full_log_path, 'saved_models'))
        with open(os.path.join(self.full_log_path, 'params.json'), 'w') as file:
            json.dump(self.params, file)
        with open(os.path.join(self.full_log_path, 'record.txt'), 'w') as file:
            file.write(starting_string + '\n')
        if self.model is not None:
            with open(os.path.join(self.full_log_path, 'model_settings.txt'), 'w') as file:
                file.write(inspect.getsource(self.model))
        else:
            with open(os.path.join(self.full_log_path, 'model_settings.txt'), 'w') as file:
                file.write(str(self.graph.as_graph_def()))

        cumulative_batch_idx = 0
        for epoch_idx in range(self.params['n_epoch']):
            generators = self.data_loader.reset_batch_generators(self.params['batch_size'])
            train_batch_generator, validate_batch_generator, test_batch_generator = generators

            cumulative_batch_idx, train_loss, train_error_rate, train_t3_error_rate = self.epoch(
                cumulative_batch_idx,
                epoch_idx,
                [
                    self.features,
                    self.labels,
                    self.keep_prob,
                    self.is_training,
                    self.lr
                ],
                [
                    self.step,
                    self.loss,
                    self.error,
                    self.t3_error,
                    self.merged
                ],
                self.sess,
                train_batch_generator,
                train_writer,
                self.params
            )
            _, validate_loss, validate_error_rate, validate_t3_error_rate = self.epoch(
                cumulative_batch_idx,
                epoch_idx,
                [
                    self.features,
                    self.labels,
                    self.keep_prob,
                    self.is_training,
                    self.lr
                ],
                [
                    self.loss,
                    self.error,
                    self.t3_error,
                    self.merged
                ],
                self.sess,
                validate_batch_generator,
                validate_writer,
                self.params
            )

            for_recording = (
                    '[%s] Epoch %d - Parameters: %s -> %s\n'
                    'train loss: %f train error: %f train top-3 error: %f '
                    'validate loss: %f validate error: %f validate top-3 error: %f' %
                    (
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        epoch_idx,
                        str(self.params),
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
                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.full_log_path,
                        'saved_models',
                        'epoch_%d' % epoch_idx,
                        self.saved_model_name
                    )
                )

            if np.isnan(train_loss):
                print(
                    'Got nan loss. Quit.'
                )
                return

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
            cum_batch_idx += 1
            if cum_batch_idx % ((epoch_idx + 1) * 10) == 0:
                log_writer.add_summary(result[-1], cum_batch_idx)
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
