#!/usr/bin/env python

import numpy as np
import tensorflow as tf

from tensorflow.contrib import cudnn_rnn


class DecoderType:
    BestPath = 0
    BeamSearch = 1


def visualize(idx, kernel):
    """
    Visualization of kernels.
    :param int idx:             Layer index
    :param tf.Variable kernel:  Filter
    """
    with tf.variable_scope('visualization'):
        x_min = tf.reduce_min(kernel)
        x_max = tf.reduce_max(kernel)
        kernel_0_to_1 = (kernel - x_min) / (x_max - x_min)
        kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])
        tf.summary.image('layer_%d' % idx, kernel_transposed, max_outputs=64)


class CTC:

    def __init__(self, img_height, img_width, char_list, max_text_len=32, decoder_type=DecoderType.BestPath):
        self.char_list = char_list
        self.max_text_len = max_text_len
        self.img_size = [img_height, img_width]
        self.decoder_type = decoder_type
        self.snapID = 0

        self.graph = None
        self.config = None
        self.sess = None

        self.features = None
        self.is_training = None
        self.seq_len = None
        self.decoder = None

    @staticmethod
    def head(x, is_training):
        with tf.name_scope('head_bn'):
            # x = tf.cast(x, tf.float32)
            # x = tf.layers.batch_normalization(
            #     x,
            #     momentum=0.9,
            #     renorm_momentum=0.9,
            #     moving_mean_initializer=tf.constant_initializer(234.0),
            #     moving_variance_initializer=tf.constant_initializer(3300.0),
            #     training=is_training
            # )
            x = tf.divide(
                tf.subtract(
                    x, 234.0
                ),
                60
            )
        # x = tf.Print(
        #     x,
        #     [
        #         tf.get_default_graph().get_tensor_by_name(
        #             [item.name for item in tf.global_variables() if 'moving_mean' in item.name][-1]
        #         ),
        #         tf.get_default_graph().get_tensor_by_name(
        #             [item.name for item in tf.global_variables() if 'moving_variance' in item.name][-1]
        #         )
        #     ]
        # )
        return x

    @staticmethod
    def bn(x, is_training):
        # with tf.name_scope('bn'):
            # x = tf.Print(x, [tf.constant('before'), x])
            # x = tf.layers.batch_normalization(
            #     x,
            #     momentum=0.9,
            #     renorm_momentum=0.9,
            #     moving_mean_initializer=tf.zeros_initializer(),
            #     moving_variance_initializer=tf.ones_initializer(),
            #     training=is_training
            # )
            # x = tf.Print(x, [tf.constant('after'), x])
        # x = tf.Print(
        #     x,
        #     [
        #         tf.get_default_graph().get_tensor_by_name(
        #             [item.name for item in tf.global_variables() if 'moving_mean' in item.name][-1]
        #         ),
        #         tf.get_default_graph().get_tensor_by_name(
        #             [item.name for item in tf.global_variables() if 'moving_variance' in item.name][-1]
        #         )
        #     ]
        # )
        return x

    def forward(self, input_imgs, is_training, keep_prob):

        # input_imgs = tf.Print(input_imgs, [tf.constant('features-before'), input_imgs, tf.reduce_mean(input_imgs),
        #                        tf.sqrt(tf.reduce_mean(tf.square(input_imgs)))])
        input_imgs = self.head(input_imgs, is_training)
        # input_imgs = tf.Print(input_imgs, [tf.constant('features-after'), input_imgs, tf.reduce_mean(input_imgs),
        #                                    tf.sqrt(tf.reduce_mean(tf.square(input_imgs)))])

        cnn_out_4d, l2_norm = self.swiss_cnn(input_imgs, is_training)
        # cnn_out_4d, l2_norm = self.experimental_cnn(input_imgs, is_training)
        # cnn_out_4d, l2_norm = self.dense_cnn(input_imgs)
        # cnn_out_4d = CTC.bn(cnn_out_4d, is_training)

        cnn_out_4d = tf.nn.dropout(cnn_out_4d, keep_prob)
        rnn_out_4d = self.swiss_rnn(cnn_out_4d)
        # rnn_out_4d = self.rnn(cnn_out_4d)

        # rnn_out_4d = tf.nn.dropout(rnn_out_4d, keep_prob)
        tail_out_3d = self.plain_tail(rnn_out_4d, len(self.char_list))

        return tail_out_3d, l2_norm

    @staticmethod
    def swiss_cnn(cnn_in_4d, is_training=None, verbose=False):
        # list of parameters for the layers
        kernel_vals = [3, 3, 3, 3, 3, 3, 2]
        feature_vals = [3, 64, 128, 256, 256, 512, 512, 512]
        stride_vals = pool_vals = [(2, 2), (2, 2), (2, 2), (2, 2), None, (2, 1), None]
        num_layers = len(stride_vals)
        conv_padding_type = [
            'SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'SAME', 'VALID'
        ]

        # create layers
        l2_norm_list = list()
        pool = cnn_in_4d  # input to first CNN layer
        for idx in range(num_layers):
            kernel = tf.Variable(
                tf.truncated_normal(
                    [
                        kernel_vals[idx],
                        kernel_vals[idx],
                        feature_vals[idx],
                        feature_vals[idx + 1]
                    ],
                    stddev=0.01
                )
            )
            if idx == 0 and (not verbose):
                visualize(0, kernel)
            conv = tf.nn.conv2d(
                pool, kernel, padding=conv_padding_type[idx], strides=(1, 1, 1, 1)
            )
            if pool_vals[idx] is None and stride_vals[idx] is None:
                conv = CTC.bn(conv, is_training)
                pool= tf.nn.relu(conv)
            else:
                relu = tf.nn.relu(conv)
                pool = tf.nn.max_pool(
                    relu,
                    (1, pool_vals[idx][0], pool_vals[idx][1], 1),
                    (1, stride_vals[idx][0], stride_vals[idx][1], 1),
                    'VALID'
                )
            if verbose:
                print(pool)
            l2_norm_list.append(
                tf.reduce_mean(tf.square(kernel))
            )

        l2_norm = tf.divide(tf.add_n(l2_norm_list), float(len(l2_norm_list)))

        return pool, l2_norm

    @staticmethod
    def experimental_cnn(cnn_in_4d, is_training, verbose=False):
        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3]
        feature_vals = [3, 32, 64, 64, 128]
        stride_vals = pool_vals = [(3, 3), (3, 2), (2, 2), (2, 2)]
        num_layers = len(stride_vals)

        # create layers
        l2_norm_list = list()
        pool = cnn_in_4d  # input to first CNN layer
        for idx in range(num_layers):
            kernel = tf.Variable(
                tf.truncated_normal(
                    [
                        kernel_vals[idx],
                        kernel_vals[idx],
                        feature_vals[idx],
                        feature_vals[idx + 1]
                    ],
                    stddev=0.01
                )
            )
            if idx == 0 and (not verbose):
                visualize(0, kernel)
            # pool = tf.Print(pool, [tf.constant(idx), tf.constant('kernel'), kernel, tf.reduce_mean(kernel), tf.sqrt(tf.reduce_mean(tf.square(kernel)))])
            # pool = tf.Print(pool, [tf.constant(idx), tf.constant('input'), pool, tf.reduce_mean(pool), tf.sqrt(tf.reduce_mean(tf.square(pool)))])
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1)
            )
            relu = tf.nn.relu(conv)
            # relu = CTC.bn(relu, is_training)
            pool = tf.nn.max_pool(
                relu,
                (1, pool_vals[idx][0], pool_vals[idx][1], 1),
                (1, stride_vals[idx][0], stride_vals[idx][1], 1),
                'VALID'
            )
            # pool = CTC.bn(pool, is_training)
            if verbose:
                print(pool)
            l2_norm_list.append(
                tf.reduce_mean(tf.square(kernel))
            )

        l2_norm = tf.divide(tf.add_n(l2_norm_list), float(len(l2_norm_list)))

        return pool, l2_norm

    @staticmethod
    def dense_cnn(cnn_in_4d, verbose=False):
        # list of parameters for the layers
        kernel_vals = [5, 5, 3, 3]
        feature_vals = [3, 32, 64, 64, 128]
        stride_vals = pool_vals = [(3, 3), (3, 2), (2, 2), (2, 2)]
        num_layers = len(stride_vals)

        # create layers
        l2_norm_list = list()
        pool = cnn_in_4d  # input to first CNN layer
        for idx in range(num_layers):
            kernel = tf.Variable(
                tf.truncated_normal(
                    [
                        kernel_vals[idx],
                        kernel_vals[idx],
                        feature_vals[idx],
                        feature_vals[idx + 1]
                    ],
                    stddev=0.1
                )
            )
            if idx == 0 and (not verbose):
                visualize(0, kernel)
            conv = tf.nn.conv2d(
                pool, kernel, padding='SAME', strides=(1, 1, 1, 1)
            )
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(
                relu,
                (1, pool_vals[idx][0], pool_vals[idx][1], 1),
                (1, stride_vals[idx][0], stride_vals[idx][1], 1),
                'VALID'
            )
            if verbose:
                print(pool)
            l2_norm_list.append(
                tf.reduce_mean(tf.square(kernel))
            )

        l2_norm = tf.divide(tf.add_n(l2_norm_list), float(len(l2_norm_list)))

        return pool, l2_norm

    @staticmethod
    def swiss_rnn(rnn_in_4d, n_cell=3):
        rnn_in_4d = tf.transpose(rnn_in_4d, [0, 2, 1, 3])
        third_dim = rnn_in_4d.shape[2]
        fourth_dim = rnn_in_4d.shape[3]
        rnn_in_3d = tf.reshape(
            rnn_in_4d,
            [
                -1,
                rnn_in_4d.shape[1],
                np.prod(
                    [
                        third_dim,
                        fourth_dim
                    ]
                )
            ]
        )

        # basic cells which is used to build RNN
        num_hidden = 256

        fw_cells = [
            cudnn_rnn.CudnnCompatibleLSTMCell(
                num_units=num_hidden,
            ) for _ in range(n_cell)
        ]
        bw_cells = [
            cudnn_rnn.CudnnCompatibleLSTMCell(
                num_units=num_hidden,
            ) for _ in range(n_cell)
        ]

        # stack basic cells
        fw_stacked = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
        bw_stacked = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_stacked,
            cell_bw=bw_stacked,
            inputs=rnn_in_3d,
            dtype=rnn_in_3d.dtype
        )

        # BxTxH + BxTxH -> BxTx2H -> Bx1xTX2H
        # result = tf.expand_dims(tf.concat([fw, bw], -1), 1)
        rnn_out_4d = tf.reshape(
            tf.concat([fw, bw], -1),
            [
                -1,
                rnn_in_4d.shape[1],
                third_dim,
                num_hidden * 2
            ]
        )
        rnn_out_4d = tf.transpose(rnn_out_4d, [0, 2, 1, 3])

        return rnn_out_4d

    @staticmethod
    def rnn(rnn_in_4d, n_cell=4, num_hidden_shrinkage=1):
        rnn_in_4d = tf.transpose(rnn_in_4d, [0, 2, 1, 3])
        third_dim = rnn_in_4d.shape[2]
        fourth_dim = rnn_in_4d.shape[3]
        rnn_in_3d = tf.reshape(
            rnn_in_4d,
            [
                -1,
                rnn_in_4d.shape[1],
                np.prod(
                    [
                        third_dim,
                        fourth_dim
                    ]
                )
            ]
        )

        # rnn_in_3d = tf.squeeze(rnn_in_4d, axis=[1])

        # basic cells which is used to build RNN
        num_hidden = int((third_dim * fourth_dim).value / num_hidden_shrinkage)
        fourth_dim = int(fourth_dim.value / num_hidden_shrinkage)

        fw_cells = [
            # tf.nn.rnn_cell.LSTMCell(
            #     num_units=num_hidden,
            #     state_is_tuple=True
            cudnn_rnn.CudnnCompatibleLSTMCell(
                num_units=num_hidden,
            ) for _ in range(n_cell)
        ]
        bw_cells = [
            # tf.nn.rnn_cell.LSTMCell(
            #     num_units=num_hidden,
            #     state_is_tuple=True
            cudnn_rnn.CudnnCompatibleLSTMCell(
                num_units=num_hidden,
            ) for _ in range(n_cell)
        ]

        # stack basic cells
        fw_stacked = tf.nn.rnn_cell.MultiRNNCell(fw_cells, state_is_tuple=True)
        bw_stacked = tf.nn.rnn_cell.MultiRNNCell(bw_cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_stacked,
            cell_bw=bw_stacked,
            inputs=rnn_in_3d,
            dtype=rnn_in_3d.dtype
        )

        # BxTxH + BxTxH -> BxTx2H -> Bx1xTX2H
        # result = tf.expand_dims(tf.concat([fw, bw], -1), 1)
        rnn_out_4d = tf.reshape(
            tf.concat([fw, bw], -1),
            [
                -1,
                rnn_in_4d.shape[1],
                third_dim,
                num_hidden * 2
            ]
        )
        rnn_out_4d = tf.transpose(rnn_out_4d, [0, 2, 1, 3])

        return rnn_out_4d

    @staticmethod
    def plain_tail(tail_in_4d, len_of_char_list):

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.truncated_normal(
                [1, 1, tail_in_4d.shape[-1].value, len_of_char_list + 1],
                stddev=0.1
            )
        )
        result = tf.squeeze(
            tf.nn.conv2d(
                tail_in_4d,
                kernel,
                [1, 1, 1, 1],
                'SAME'
            ),
            axis=[1]
        )
        return result

    @staticmethod
    def fc_tail(tail_in_4d, len_of_char_list):
        kernel = tf.Variable(
            tf.truncated_normal(
                [1, 1, tail_in_4d.shape[-1].value, int(tail_in_4d.shape[-1].value / 2)],
                stddev=0.1
            )
        )
        # bias = tf.Variable(
        #     tf.constant(
        #         0.1,
        #         shape=[int(tail_in_4d.shape[-1].value / 2)]
        #     )
        # )
        bias = tf.Variable(
            tf.truncated_normal(
                [int(tail_in_4d.shape[-1].value / 2)],
                stddev=0.1
            )
        )
        first_layer_output = tf.nn.elu(
            tf.add(
                tf.nn.conv2d(
                    tail_in_4d,
                    kernel,
                    [1, 1, 1, 1],
                    'SAME'
                ),
                bias
            )
        )

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(
            tf.truncated_normal(
                [1, 1, int(tail_in_4d.shape[-1].value / 2), len_of_char_list + 1],
                stddev=0.1
            )
        )
        result = tf.squeeze(
            tf.nn.conv2d(
                first_layer_output,
                kernel,
                [1, 1, 1, 1],
                'SAME'
            ),
            axis=[1]
        )
        return result

    def setup_decoder(self, ctc_in_3d, seq_len):
        # BxTxC -> TxBxC
        ctc_in_3d_tbc = tf.transpose(ctc_in_3d, [1, 0, 2])

        # decoder: either best path decoding or beam search decoding
        if self.decoder_type == DecoderType.BestPath:
            decoder = tf.nn.ctc_greedy_decoder(
                inputs=ctc_in_3d_tbc,
                sequence_length=seq_len
            )
        elif self.decoder_type == DecoderType.BeamSearch:
            decoder = tf.nn.ctc_beam_search_decoder(
                inputs=ctc_in_3d_tbc,
                sequence_length=seq_len,
                beam_width=50,
                merge_repeated=False
            )
        else:
            raise ValueError(
                'Unknown decoder type: %s.' %
                self.decoder_type
            )

        return decoder

    def setup_tf(self):
        # TODO: not used or tested
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(
            graph=self.graph, config=self.config
        )

        with self.graph.as_default():
            saver = tf.train.Saver()
        model_dir = '../model/'
        latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(self.sess, latest_snapshot)
        else:
            print('Init with new values')
            self.sess.run(tf.global_variables_initializer())

        self.features = self.graph.get_tensor_by_name('features:0')
        self.is_training = self.graph.get_tensor_by_name('is_training:0')
        logits = self.graph.get_tensor_by_name('logits:0')
        self.seq_len = self.graph.get_tensor_by_name('seq_len:0')
        self.decoder = self.setup_decoder(logits, self.seq_len)

    def decoder_output_to_text(self, ctc_output):

        batch_size = ctc_output.shape[0]

        # contains string of labels for each batch element
        encoded_label_strs = [[] for _ in range(batch_size)]

        decoded = ctc_output[0][0]

        # go over all indices and save mapping: batch -> values
        # TODO: what does this do?
        idxDict = {b: [] for b in range(batch_size)}
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batch_element = idx2d[0]  # index according to [b,t]
            encoded_label_strs[batch_element].append(label)

        result = [
            str().join(
                [
                    self.char_list[c]
                    for c in labelStr]
            )
            for labelStr in encoded_label_strs
        ]

        return result

    def infer_batch(self, batch):
        decoded = self.sess.run(
            self.decoder,
            {
                self.features: batch.imgs,
                self.seq_len: [self.max_text_len] * batch.shape[0],
                self.is_training: False
            }
        )
        return self.decoder_output_to_text(decoded)


if __name__ == '__main__':

    from chrecog.loaders import WordsSet
    ws = WordsSet()

    len_of_char_list = len(ws.get_characters())
    height, width, channel = ws.get_img_shape()
    batch_size = 8

    cnn_in_4d = batch_img = tf.random_normal([batch_size, height, width, channel])

    cnn_o, _ = CTC.swiss_cnn(batch_img, False, verbose=True)

    rnn_o = CTC.swiss_rnn(cnn_o)
    print(rnn_o)

    tail_o = CTC.plain_tail(rnn_o, len_of_char_list)
    print(tail_o)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    #
    # sess.run(tf.global_variables_initializer())
    #
    # sess.run(cnn_o)
    # sess.run(rnn_o)
    # sess.run(tail_o)
