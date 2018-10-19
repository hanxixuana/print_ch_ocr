#!/usr/bin/env python

import os
import cv2
import numpy as np
import tensorflow as tf

from copy import deepcopy
from tensorflow.python.saved_model import tag_constants


class CharRecogEngine(object):
    """

    import os
    import numpy as np
    import tensorflow as tf

    from copy import deepcopy
    from chrecog.loaders import AugOCRSet
    from tensorflow.python.saved_model import tag_constants

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    number = 10
    loader = AugOCRSet()
    samples, labels = loader.get_random_samples(number)
    batch_image = samples
    mapper = loader.label_to_char_mapper

    path_to_saved_model = 'D:\projects\ch_ocr\logs\log_2018_09_24_16_26_36_510687\saved_models\epoch_190'
    class fool: pass
    self = fool()

    """
    def __init__(self, path=None, mapper=None):

        self.class_idx_to_label_mapper = deepcopy(mapper)
        self.path_to_saved_model = path

        self.graph = None
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = None

        self.features = None
        self.keep_prob = None
        self.is_training = None
        self.logits = None

        if self.path_to_saved_model is not None:
            self.load_saved_model(self.path_to_saved_model)

    def load_saved_model(self, path_to_saved_model):

        self.path_to_saved_model = path_to_saved_model
        files_in_folder = os.listdir(self.path_to_saved_model)
        meta_files = [item for item in files_in_folder if '.meta' in item]
        pb_files = [item for item in files_in_folder if '.pb' in item]
        if len(meta_files) + len(pb_files) == 0:
            raise FileNotFoundError(
                'Cannot find *.pb or *.meta in the folder pointed to by %s' %
                self.path_to_saved_model
            )
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.config)
        if len(pb_files) > 0:
            tf.saved_model.loader.load(
                self.sess,
                [tag_constants.SERVING],
                self.path_to_saved_model
            )
        elif len(meta_files) > 0:
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(
                    os.path.join(
                        self.path_to_saved_model,
                        meta_files[0]
                    )
                )
            saver.restore(
                self.sess,
                tf.train.latest_checkpoint(self.path_to_saved_model)
            )
        self.features = self.graph.get_tensor_by_name('features:0')
        self.keep_prob = self.graph.get_tensor_by_name('keep_prob:0')
        self.is_training = self.graph.get_tensor_by_name('is_training:0')
        self.logits = self.graph.get_tensor_by_name('logits:0')

    @staticmethod
    def check_batch_image(batch_image):
        if not isinstance(batch_image, np.ndarray):
            raise TypeError(
                'batch_image should be of the type np.ndarray.'
            )
        if len(batch_image.shape) != 4:
            raise ValueError(
                'batch_image should have 4 dimensions (batch x height x width x channel).'
            )
        if batch_image.shape[-1] != 3:
            raise ValueError(
                'batch_image should have 3 channels (red, green, blue).'
            )
        if batch_image.dtype != 'float32':
            batch_image = batch_image.astype('float32')
        return batch_image

    def predict(self, batch_image, top_n=1):

        self.check_batch_image(batch_image)

        logits = self.sess.run(
            self.logits,
            feed_dict={
                self.features: batch_image,
                self.keep_prob: 1.0,
                self.is_training: False
            }
        )
        if len(logits.shape) == 4:
            labels = np.argsort(logits, -1)
            labels = labels[:, :, :, ::-1][:, :, :, :top_n]
            logits = np.sort(logits, -1)
            logits = logits[:, :, :, ::-1]
            likelihood = np.exp(logits) / (np.sum(np.exp(logits), axis=-1)[:, :, :, np.newaxis] + 1e-6)
            likelihood = likelihood[:, :, :, :top_n]
        elif len(logits.shape) == 2:
            labels = np.argsort(logits, -1)
            labels = labels[:, ::-1][:, :top_n]
            logits = np.sort(logits, -1)
            logits = logits[:, ::-1]
            likelihood = np.exp(logits) / (np.sum(np.exp(logits), axis=-1)[:, np.newaxis] + 1e-6)
            likelihood = likelihood[:, :top_n]
        else:
            raise ValueError(
                'logits has a shape of (%s). Don\'t know how to deal with it.' %
                ', '.join([str(item) for item in logits.shape])
            )

        if self.class_idx_to_label_mapper is not None:
            labels = np.vectorize(self.class_idx_to_label_mapper.get)(labels)

        return labels, likelihood

    @staticmethod
    def check_image(image):
        if not isinstance(image, np.ndarray):
            raise TypeError(
                'batch_image should be of the type np.ndarray.'
            )
        if len(image.shape) != 3:
            raise ValueError(
                'batch_image should have 3 dimensions (height x width x channel).'
            )
        if image.shape[-1] != 3:
            raise ValueError(
                'batch_image should have 3 channels (red, green, blue).'
            )
        if image.dtype != 'float32':
            image = image.astype('float32')
        if image.shape[0] < 64:
            image = cv2.resize(
                image, None, fx=64/image.shape[0], fy=64/image.shape[0]
            )
        return image

    def scan(self, image, top_n=1):

        image = self.check_image(image)

        logits = self.sess.run(
            self.logits,
            feed_dict={
                self.features: image[np.newaxis, :, :, :],
                self.keep_prob: 1.0,
                self.is_training: False
            }
        )
        if len(logits.shape) == 4:
            labels = np.argsort(logits, -1)
            labels = labels[:, :, :, ::-1][:, :, :, :top_n]
            logits = np.sort(logits, -1)
            logits = logits[:, :, :, ::-1]
            likelihood = (
                    np.exp(logits[:, :, :, :top_n])
                    /
                    (np.sum(np.exp(logits), axis=-1)[:, :, :, np.newaxis] + 1e-8)
            )
        elif len(logits.shape) == 2:
            labels = np.argsort(logits, -1)
            labels = labels[:, ::-1][:, :top_n]
            logits = np.sort(logits, -1)
            logits = logits[:, ::-1]
            likelihood = (
                    np.exp(logits[:, :top_n])
                    /
                    (np.sum(np.exp(logits), axis=-1)[:, np.newaxis] + 1e-8)
            )
        else:
            raise ValueError(
                'logits has a shape of (%s). Don\'t know how to deal with it.' %
                ', '.join([str(item) for item in logits.shape])
            )

        if self.class_idx_to_label_mapper is not None:
            labels = np.vectorize(self.class_idx_to_label_mapper.get)(labels)

        return labels[0], likelihood[0]


if __name__ == '__main__':

    from chrecog.loaders import AugOCRSet

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    number = 10
    path = 'D:\projects\ch_ocr\logs\log_2018_09_24_16_26_36_510687\saved_models\epoch_190'

    loader = AugOCRSet()
    rand_batch, batch_label = loader.get_random_samples(number)
    rand_str, str_label = loader.get_random_string(number)

    model = CharRecogEngine(path, loader.label_to_char_mapper)

    pred_batch, batch_lik = model.predict(rand_batch, top_n=3)
    pred_str, str_lik = model.scan(rand_str, top_n=1)

