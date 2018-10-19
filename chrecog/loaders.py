#!/usr/bin/env python

import os
import json
import h5py
import numpy as np

from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split
from abc import ABCMeta, abstractmethod


class DataSet(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def split_data_into_tvt(self, tvt_propositions):
        raise NotImplementedError

    @abstractmethod
    def reset_batch_generators(self, batch_size):
        raise NotImplementedError


class WordsSet(DataSet):
    def __init__(self, path_to_img_set='D:\\datasets\\more_words\\words.h5'):
        super(WordsSet, self).__init__()

        self.full_path_to_img_set = os.path.abspath(
            path_to_img_set
        )
        self.data_set = h5py.File(self.full_path_to_img_set)
        self.data_set_in_memory = None

        self.train_sample_idx_list = list()
        self.validate_sample_idx_list = list()
        self.test_sample_idx_list = list()

        self.train_batch_generator = None
        self.validate_batch_generator = None
        self.test_batch_generator = None

    def get_random_sample(self):
        idx = np.random.choice(self.data_set['n_imgs'].value, 1)[0]
        label = self.data_set['labels'][idx]
        sample = self.data_set[str(idx)].value
        return sample, label

    def get_img_shape(self):
        return (
            self.data_set['0'].value.shape
        )

    def get_max_test_len(self):
        return np.max(
            [
                len(item)
                for item in self.data_set['labels']
            ]
        )

    def get_characters(self):
        result = [item.decode('utf8') for item in self.data_set['characters'].value.tolist()]
        return result

    def get_n_imgs(self):
        return self.data_set['n_imgs'].value

    def split_data_into_tvt(self, tvt_propositions=(0.6, 0.3, 0.1), load_to_memory=True):

        keys = list(self.data_set.keys())
        keys.remove('n_imgs')
        keys.remove('labels')
        keys.remove('characters')

        data_set_key_list = list(
            [
                [
                    self.data_set['labels'][int(key)],
                    key
                ]
                for key in keys
            ]
        )

        train_and_validate, self.test_sample_idx_list = train_test_split(
            data_set_key_list,
            test_size=tvt_propositions[2]
        )
        self.train_sample_idx_list, self.validate_sample_idx_list = train_test_split(
            train_and_validate,
            test_size=tvt_propositions[1]
        )

        if load_to_memory:
            self.data_set_in_memory = dict()
            for key in tqdm(keys):
                self.data_set_in_memory[key] = self.data_set[key].value

    @staticmethod
    def h5_batch_generator_maker(idx_list, batch_size, data_set):
        n_idx = len(idx_list)
        n_batch = int(n_idx / batch_size)
        for batch_idx in range(n_batch):
            batch_idx_list = idx_list[(batch_idx * batch_size):(batch_idx * batch_size + batch_size)]
            batch_img_list = list()
            batch_label_list = list()
            for label, h5_ds_idx in batch_idx_list:
                batch_img_list.append(
                    data_set[h5_ds_idx].value
                )
                batch_label_list.append(
                    label.decode('utf8')
                )
            batch_img = np.stack(batch_img_list, axis=0)
            batch_label = np.array(batch_label_list)
            yield batch_idx, batch_img, batch_label

    @staticmethod
    def np_batch_generator_maker(idx_list, batch_size, data_set):
        n_idx = len(idx_list)
        n_batch = int(n_idx / batch_size)
        for batch_idx in range(n_batch):
            batch_idx_list = idx_list[(batch_idx * batch_size):(batch_idx * batch_size + batch_size)]
            batch_img_list = list()
            batch_label_list = list()
            for label, h5_ds_idx in batch_idx_list:
                batch_img_list.append(
                    data_set[h5_ds_idx]
                )
                batch_label_list.append(
                    label.decode('utf8')
                )
            batch_img = np.stack(batch_img_list, axis=0)
            batch_label = np.array(batch_label_list)
            yield batch_idx, batch_img, batch_label

    def reset_batch_generators(self, batch_size):
        np.random.shuffle(self.train_sample_idx_list)
        np.random.shuffle(self.validate_sample_idx_list)
        np.random.shuffle(self.test_sample_idx_list)
        if self.data_set_in_memory is None:
            self.train_batch_generator = self.h5_batch_generator_maker(
                self.train_sample_idx_list,
                batch_size,
                self.data_set
            )
            self.validate_batch_generator = self.h5_batch_generator_maker(
                self.validate_sample_idx_list,
                batch_size,
                self.data_set
            )
            self.test_batch_generator = self.h5_batch_generator_maker(
                self.test_sample_idx_list,
                batch_size,
                self.data_set
            )
        else:
            self.train_batch_generator = self.np_batch_generator_maker(
                self.train_sample_idx_list,
                batch_size,
                self.data_set_in_memory
            )
            self.validate_batch_generator = self.np_batch_generator_maker(
                self.validate_sample_idx_list,
                batch_size,
                self.data_set_in_memory
            )
            self.test_batch_generator = self.np_batch_generator_maker(
                self.test_sample_idx_list,
                batch_size,
                self.data_set_in_memory
            )
        return (
            self.train_batch_generator,
            self.validate_batch_generator,
            self.test_batch_generator
        )


class AugOCRSet(DataSet):
    def __init__(self, path_to_img_set='D:\\datasets\\augmented_ocr_8\\tiny_augmented_imgs\\processed_data'):
        super(AugOCRSet, self).__init__()

        self.full_path_to_img_set = os.path.abspath(
            path_to_img_set
        )
        with open(os.path.join(self.full_path_to_img_set, 'summary.json')) as file_handle:
            self.summary = json.load(file_handle)
        self.char_to_h5_mapper = {
            key: h5py.File(
                os.path.join(
                    self.full_path_to_img_set,
                    self.summary[key][0]
                )
            )
            for key in self.summary
        }
        self.char_to_label_mapper = dict()
        for idx, character in enumerate(self.char_to_h5_mapper):
            self.char_to_label_mapper[character] = idx

        self.label_to_char_mapper = dict()
        for character in self.char_to_label_mapper:
            self.label_to_char_mapper[self.char_to_label_mapper[character]] = character

        self.train_sample_idx_list = list()
        self.validate_sample_idx_list = list()
        self.test_sample_idx_list = list()

        self.train_batch_generator = None
        self.validate_batch_generator = None
        self.test_batch_generator = None

    def get_random_samples(self, number):
        samples = list()
        labels = list()
        all_labels = list(self.char_to_label_mapper.keys())
        for idx in range(number):
            label = np.random.choice(all_labels, 1)[0]
            labels.append(label)
            samples.append(
                self.get_samples(
                    label,
                    1
                )
            )
        samples = np.concatenate(samples)
        return samples, labels

    def get_random_string(self, number):
        samples, labels = self.get_random_samples(number)
        samples = np.hstack([item[0] for item in np.split(samples, indices_or_sections=samples.shape[0])])
        return samples, labels

    def get_samples(self, character, number):
        try:
            data_set_key_list = list(self.char_to_h5_mapper[character].keys())
        except KeyError:
            raise KeyError(
                '[%s] Character %s is not contained in the data set.' %
                (
                    datetime.now().strftime('%Y-%m-%D %H:%M:%S.%F'),
                    character
                )
            )
        data_set_key_list.remove('label')
        if number > len(data_set_key_list):
            raise ValueError(
                '[%s] Cannot get %d samples, because there are only %d of %s.' %
                (
                    datetime.now().strftime('%Y-%m-%D %H:%M:%S.%F'),
                    number,
                    len(data_set_key_list),
                    character
                )
            )
        result = list()
        for key in np.random.choice(data_set_key_list, number, replace=False):
            result.append(
                self.char_to_h5_mapper[character][key][:]
            )
        result = np.stack(result, axis=0)
        return result

    def get_img_shape(self):
        return (
            self.char_to_h5_mapper[list(self.char_to_label_mapper.keys())[0]]['0'][:].shape
        )

    def get_n_labels(self):
        return len(self.char_to_label_mapper)

    def split_data_into_tvt(self, tvt_propositions=(0.6, 0.3, 0.1)):
        self.train_sample_idx_list = list()
        self.validate_sample_idx_list = list()
        self.test_sample_idx_list = list()

        for character in tqdm(self.char_to_h5_mapper):
            data_set_key_list = list(self.char_to_h5_mapper[character].keys())
            data_set_key_list.remove('label')
            train_and_validate, test_set_key_list = train_test_split(
                data_set_key_list,
                test_size=tvt_propositions[2]
            )
            train_set_key_list, validate_set_list = train_test_split(
                train_and_validate,
                test_size=tvt_propositions[1]
            )
            self.train_sample_idx_list.extend(
                [(character, item) for item in train_set_key_list]
            )
            self.validate_sample_idx_list.extend(
                [(character, item) for item in validate_set_list]
            )
            self.test_sample_idx_list.extend(
                [(character, item) for item in test_set_key_list]
            )

    @staticmethod
    def batch_generator_maker(idx_list, batch_size, char_to_h5_mapper, char_to_label_mapper):
        n_idx = len(idx_list)
        n_batch = int(n_idx / batch_size)
        for batch_idx in range(n_batch):
            batch_idx_list = idx_list[(batch_idx * batch_size):(batch_idx * batch_size + batch_size)]
            batch_img_list = list()
            batch_label_list = list()
            for character, h5_ds_idx in batch_idx_list:
                batch_img_list.append(
                    char_to_h5_mapper[character][h5_ds_idx][:]
                )
                batch_label_list.append(
                    char_to_label_mapper[character]
                )
            batch_img = np.stack(batch_img_list, axis=0)
            batch_label = np.array(batch_label_list)
            yield batch_idx, batch_img, batch_label

    def reset_batch_generators(self, batch_size):
        np.random.shuffle(self.train_sample_idx_list)
        np.random.shuffle(self.validate_sample_idx_list)
        np.random.shuffle(self.test_sample_idx_list)
        self.train_batch_generator = self.batch_generator_maker(
            self.train_sample_idx_list,
            batch_size,
            self.char_to_h5_mapper,
            self.char_to_label_mapper
        )
        self.validate_batch_generator = self.batch_generator_maker(
            self.validate_sample_idx_list,
            batch_size,
            self.char_to_h5_mapper,
            self.char_to_label_mapper
        )
        self.test_batch_generator = self.batch_generator_maker(
            self.test_sample_idx_list,
            batch_size,
            self.char_to_h5_mapper,
            self.char_to_label_mapper
        )
        return (
            self.train_batch_generator,
            self.validate_batch_generator,
            self.test_batch_generator
        )
