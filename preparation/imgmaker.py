#!/usr/bin/env python

import os
import cv2
import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pandas.errors import ParserError
from datetime import datetime
import pathos.pools as pp

from preparation.augmentation import soft_seq


class DataSet(object):
    # full_path_of_root_folder = 'D:\\datasets\\all_raw_ocr'
    full_path_of_root_folder = 'D:\\datasets\\raw_ocr'
    # sub_folders = (
    #     'B1', 'B10', 'B11', 'B12', 'B13', 'B17', 'B18', 'B19', 'B2', 'B20', 'B21', 'B22', 'B26',
    #     'B27', 'B29', 'B3', 'B30', 'B31', 'B32', 'B35', 'B4', 'B42', 'B5', 'B6', 'B7', 'B8', 'B9'
    # )
    sub_folders = ['B26', 'B27', 'B29', 'B30', 'B31', 'B32', 'B35', 'B42']

    def __init__(self):
        self.__full_paths_of_sub_folders = [
            os.path.join(self.full_path_of_root_folder, sub_folder)
            for sub_folder in self.sub_folders
        ]
        self.__current_sub_folder = None
        self.__full_path_of_current_sub_folder = None

        self.__orgbox_images = list()
        self.__original_images = list()
        self.__orgbox_csvs = list()

        self.full_path_of_current_orgbox_image = None
        self.full_path_of_current_original_image = None
        self.full_path_of_current_csv = None

        self.img = None
        self.orig_img = None
        self.table = None
        self.correct = None

        self.total_result = dict()
        self.img_count_before_augmentation = dict()

    def full_paths_of_sub_folders(self):
        return self.__full_paths_of_sub_folders

    def total_n_of_sub_folders(self):
        return len(self.sub_folders)

    def get_sub_folder(self, idx):
        self.__current_sub_folder = self.sub_folders[idx]
        self.__full_path_of_current_sub_folder = os.path.join(
            self.full_path_of_root_folder, self.__current_sub_folder
        )
        self.__orgbox_images = [
            os.path.join(self.__full_path_of_current_sub_folder, path)
            for path in os.listdir(self.__full_path_of_current_sub_folder)
            if path[:7] == 'orgbox-'
        ]
        self.__original_images = [
            os.path.join(self.__full_path_of_current_sub_folder, path)
            for path in os.listdir(self.__full_path_of_current_sub_folder)
            if path[:4] == 'org-'
        ]
        self.__orgbox_csvs = [
            os.path.join(self.__full_path_of_current_sub_folder, path)
            for path in os.listdir(self.__full_path_of_current_sub_folder)
            if path[:4] == 'out-'
        ]
        return self.__orgbox_images, self.__orgbox_csvs

    def total_n_of_orgbox_images_and_csvs(self):
        if (
                (len(self.__orgbox_csvs) != len(self.__orgbox_images))
                or
                (len(self.__orgbox_csvs) != len(self.__original_images))
        ):
            raise ValueError(
                'Number of orgbox images %d != that of out csvs %d != that of original images %d.' %
                (
                    len(self.__orgbox_images),
                    len(self.__orgbox_csvs),
                    len(self.__original_images)
                )
            )
        else:
            return len(self.__orgbox_csvs)

    @staticmethod
    def find_nth_occurence(haystack, needle, n):
        parts = haystack.split(needle, n + 1)
        if len(parts) <= n + 1:
            return -1
        return len(haystack) - len(parts[-1]) - len(needle)

    def remove_nth_char(self, haystack, ch, n):
        where = self.find_nth_occurence(
            haystack, ch, n
        )
        return (
                haystack[:where]
                +
                haystack[(where+1):]
        )

    def get_an_orgbox_image_and_csv(self, idx):
        self.full_path_of_current_orgbox_image = self.__orgbox_images[idx]
        self.full_path_of_current_original_image = self.__original_images[idx]
        self.full_path_of_current_csv = self.__orgbox_csvs[idx]

        self.img = cv2.imread(self.full_path_of_current_orgbox_image)
        self.orig_img = cv2.imread(self.full_path_of_current_original_image)
        try:
            self.table = pd.read_csv(self.full_path_of_current_csv, sep=',')
        except ParserError as error:
            print(
                'Cannot read the csv %s in its current status: %s. '
                'Trying to decorate it and reopen it.' %
                (
                    self.full_path_of_current_csv,
                    error
                )
            )
            raise error
            # with open(self.full_path_of_current_csv, 'r') as file:
            #
            #     file = open(self.full_path_of_current_csv, 'r', encoding='utf-8-sig')
            #     text_list = file.read().split('\n')
            #     start_comma_idx = text_list[0].split(',').index('conf')
            #
            #     for idx, text in enumerate(text_list):
            #         idx = 6
            #         text = text_list[idx]
            #         if len(text) > 0:
            #             text_start_idx = self.find_nth_occurence(text, ',', start_comma_idx) + 1
            #             text_end_idx = len(text) - text[::-1].index(',') - 1
            #             text_of_interest = text[text_start_idx:text_end_idx]
            #             if text_of_interest.count(',') > 3:
            #                 del text_list[idx]
            #             elif text_of_interest.count(',') == 2:
            #                 text_of_interest_len_list = [
            #                     len(item) for item in text_of_interest.split(',')
            #                 ]
            #                 left_two_len = (
            #                         text_of_interest_len_list[0]
            #                         +
            #                         text_of_interest_len_list[1]
            #                 )
            #                 right_two_len = (
            #                         text_of_interest_len_list[1]
            #                         +
            #                         text_of_interest_len_list[2]
            #                 )
            #                 if left_two_len >= text_of_interest_len_list[2]:
            #                     if left_two_len - text_of_interest_len_list[2] <= 2:
            #                         text_list[idx] = (
            #                                 text_list[idx][:text_start_idx]
            #                                 +
            #                                 self.remove_nth_char(text_of_interest, ',', 1)
            #                                 +
            #                                 text_list[idx][text_end_idx:]
            #                         )
            #                     else:
            #                         del text_list[idx]
            #                 elif right_two_len >= text_of_interest_len_list[0]:
            #                     if right_two_len - text_of_interest_len_list[0] <= 2:
            #                         text_list[idx] = (
            #                                 text_list[idx][:text_start_idx]
            #                                 +
            #                                 self.remove_nth_char(text_of_interest, ',', 0)
            #                                 +
            #                                 text_list[idx][text_end_idx:]
            #                         )
            #                     else:
            #                         del text_list[idx]
            #                 else:
            #                     # should never reach here
            #                     del text_list[idx]

        if len(self.table.keys()) == 1:
            self.table = pd.read_csv(self.full_path_of_current_csv, sep=';')
        self.correct = self.table[
            (
                    self.table['textno'].notnull()
                    &
                    self.table['label'].notnull()
            )
        ]
        return self.img, self.table

    def get_total_n_of_boxes(self, only_use_correct_ones=True):
        if only_use_correct_ones:
            if self.correct is None:
                if self.table is not None:
                    self.correct = self.table[
                        (
                            self.table['textno'].notnull()
                            &
                            self.table['label'].notnull()
                        )
                    ]
                    return len(self.correct)
                else:
                    return 0
            else:
                return len(self.correct)
        else:
            return (
                len(self.table)
                if self.table is not None
                else 0
            )

    def get_a_box_and_title(self, idx, only_use_correct_ones=True, show_img=False):
        if only_use_correct_ones:
            if self.correct is None:
                self.correct = self.table[
                    (
                        self.table['textno'].notnull()
                        &
                        self.table['label'].notnull()
                    )
                ]
            left, top, width, height = (
                self.correct[
                    ['left', 'top', 'width', 'height']
                ].iloc[idx]
            )
            text, label = (
                self.correct[
                    ['text', 'label']
                ].iloc[idx]
            )
            cropped_image = self.orig_img[top:(top + height), left:(left + width)]
        else:
            left, top, width, height = (
                self.table[
                    ['left', 'top', 'width', 'height']
                ].iloc[idx]
            )
            text, label = (
                self.table[
                    ['text', 'label']
                ].iloc[idx]
            )
            cropped_image = self.orig_img[top:(top + height), left:(left + width)]
        if show_img:
            fig = plt.figure()
            plt.imshow(cropped_image)
            fig.suptitle('text: %s - label: %s' % (text, label))
            fig.show()
        return cropped_image, text, label

    def show_n_squared_of_boxes(self, n, figsize=(16, 12), only_use_correct_ones=True):
        img_idx_list = np.random.randint(
            0, self.get_total_n_of_boxes(only_use_correct_ones=only_use_correct_ones), n ** 2
        )
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(nrows=n, ncols=n)
        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                img, text, label = self.get_a_box_and_title(
                    img_idx_list[col_idx + row_idx * n],
                    only_use_correct_ones=only_use_correct_ones
                )
                ax.imshow(img)
                if only_use_correct_ones:
                    ax.set_title(
                        'textno: %d text: %s label: %s' %
                        (
                            self.correct['textno'].iloc[img_idx_list[col_idx + row_idx * n]],
                            text,
                            label
                        )
                    )
                else:
                    ax.set_title(
                        'textno: %d text: %s label: %s' %
                        (
                            self.table['textno'][img_idx_list[col_idx + row_idx * n]],
                            text,
                            label
                        )
                    )
        fig.suptitle(
            self.full_path_of_current_orgbox_image
            +
            '\n'
            +
            self.full_path_of_current_original_image
            +
            '\n'
            +
            self.full_path_of_current_csv
        )
        fig.show()

    def get_a_box_and_title_by_textno(self, textno, show_img=False):
        index_series = self.table['textno'] == textno
        left, top, width, height = (
            self.table[index_series][
                ['left', 'top', 'width', 'height']
            ].iloc[0]
        )
        text, label = (
            self.table[index_series][
                ['text', 'label']
            ].iloc[0]
        )
        cropped_image = self.orig_img[top:(top + height), left:(left + width)]
        if show_img:
            fig = plt.figure()
            plt.imshow(cropped_image)
            fig.suptitle('text: %s - label: %s' % (text, label))
            fig.show()
        return cropped_image, text, label

    def split_box_into_chars(self, idx, show_img=False):

        img, text, label = self.get_a_box_and_title(idx)

        if (
                (
                        img.shape[0] > 500
                        and
                        img.shape[1] > 500
                )
                or
                img.shape[0] == 0
                or
                img.shape[1] == 0
        ):
            print(
                '===========\n'
                '%s\n%s\n%s\n%s\n%s\n'
                '===========\n' %
                (
                    self.full_path_of_current_orgbox_image,
                    self.full_path_of_current_original_image,
                    self.full_path_of_current_csv,
                    (
                            str(self.correct.index[idx])
                            +
                            ' -> '
                            +
                            ': '.join(
                                repr(self.correct.iloc[idx]).replace('\n', ' ').split()
                            )
                    ),
                    'The box in this row may have a wrong shape leading to a box of the shape of %s.' % str(img.shape)
                )
            )
            return dict()
        if len(np.unique(img)) <= 1:
            print(
                '===========\n'
                '%s\n%s\n%s\n%s\n%s\n'
                '===========\n' %
                (
                    self.full_path_of_current_orgbox_image,
                    self.full_path_of_current_original_image,
                    self.full_path_of_current_csv,
                    (
                            str(self.correct.index[idx])
                            +
                            ' -> '
                            +
                            ': '.join(
                                repr(self.correct.iloc[idx]).replace('\n', '; ').split()
                            )
                    ),
                    'The box in this row may contain nothing.'
                )
            )
            return dict()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.bitwise_not(gray)
        gray = cv2.adaptiveThreshold(gray, gray.max(), cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)

        mountains = (gray / gray.max()).sum(0).astype(int)

        valley_check = mountains == 0
        valleys = list()
        for idx in range(len(valley_check) - 1):
            if valley_check[idx] < valley_check[idx + 1]:
                valleys.append([idx + 1])
            elif valley_check[idx] > valley_check[idx + 1]:
                if len(valleys) > 0:
                    valleys[-1].append(idx)
        if len(valleys) > 0:
            if len(valleys[0]) < 2:
                valleys.pop(0)
        if len(valleys) > 0:
            if len(valleys[-1]) < 2:
                valleys.pop(-1)

        valley_bottom = [np.round(np.mean(item)) for item in valleys]

        if show_img:
            plt.figure()
            plt.imshow(gray)
            for place in valley_bottom:
                plt.plot([place] * gray.shape[0], range(gray.shape[0]))
            plt.show()

        result = dict()
        if len(valley_bottom) + 1 == len(label):
            start = 0
            for idx, end in enumerate(valley_bottom + [img.shape[1]]):
                if label[idx] in result:
                    result[label[idx]].append(
                        img[:, int(start):int(end)]
                    )
                else:
                    result[label[idx]] = [
                        img[:, int(start):int(end)]
                    ]
                start = end

        return result

    def split_all_correct_boxes(self, img_idx):
        result = dict()
        self.get_an_orgbox_image_and_csv(img_idx)
        for idx in range(self.get_total_n_of_boxes()):
            box_result = self.split_box_into_chars(idx)
            for key in box_result:
                if key in result:
                    result[key].extend(box_result[key])
                else:
                    result[key] = box_result[key]
        return result

    def split_sub_folder(self, sub_folder_idx):
        result = dict()
        self.get_sub_folder(sub_folder_idx)
        for idx in range(self.total_n_of_orgbox_images_and_csvs()):
            img_result = self.split_all_correct_boxes(idx)
            for key in img_result:
                if key in result:
                    result[key].extend(img_result[key])
                else:
                    result[key] = img_result[key]
        return result

    def split_all_sub_folders(self, max_height=None, max_width=None, zero_padding_coef=1.0):
        self.total_result = dict()
        for idx in tqdm(range(len(self.sub_folders))):
            sub_folder_result = self.split_sub_folder(idx)
            for key in sub_folder_result:
                if key in self.total_result:
                    self.total_result[key].extend(sub_folder_result[key])
                else:
                    self.total_result[key] = sub_folder_result[key]
        if max_height is None:
            max_height = max(
                [
                    max(
                        item.shape[0]
                        for item in self.total_result[key]
                    )
                    for key in self.total_result
                ]
            )

        if max_width is None:
            max_width = max(
                [
                    max(
                        item.shape[1]
                        for item in self.total_result[key]
                    )
                    for key in self.total_result
                ]
            )
        self.total_result[' '] = [
            np.ones([max_height, max_width, 3], dtype='uint8') * 255
        ]
        self.total_result.pop('©', None)

        for key in tqdm(self.total_result):
            for idx, img in enumerate(self.total_result[key]):
                height, width, _ = img.shape
                if height > max_height or width > max_width:
                    scaling_coef = (
                            min(
                                [
                                    max_height / height,
                                    max_width / width
                                ]
                            )
                            *
                            zero_padding_coef
                    )
                    temp = cv2.resize(
                        img,
                        None,
                        fx=scaling_coef,
                        fy=scaling_coef
                    )
                else:
                    temp = self.total_result[key][idx]
                up_padding_length = int((max_height - temp.shape[0]) / 2)
                down_padding_length = int(max_height - up_padding_length - temp.shape[0])
                left_padding_length = int((max_width - temp.shape[1]) / 2)
                right_padding_length = int(max_width - left_padding_length - temp.shape[1])

                self.total_result[key][idx] = cv2.copyMakeBorder(
                    temp,
                    up_padding_length,
                    down_padding_length,
                    left_padding_length,
                    right_padding_length,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )

        self.img_count_before_augmentation = {
            key: len(self.total_result[key])
            for key in self.total_result
        }

        if not os.path.exists('./processed_data'):
            os.makedirs('./processed_data')

        self.data_augmentation(total_n_img_per_class_coef=1.2)
        self.save_raw_data()

        return self.total_result

    def save_raw_data(self):
        path_to_file = (
                './processed_data/raw_data_%s.npy' %
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        np.save(path_to_file, self.total_result)
        print(
            '[%s] Saved the raw data to %s.' %
            (
                datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f'),
                path_to_file
            )
        )

    def data_augmentation(self, total_n_img_per_class_coef=1.0):
        max_n_img_per_char = int(
                max(
                    len(self.total_result[key])
                    for key in self.total_result
                )
                *
                total_n_img_per_class_coef
        )

        def do(label, current_img_list, max_n_img):
            new_img_list = list()
            current_n_img = len(current_img_list)
            for new_img_idx in range(max_n_img - current_n_img):
                new_img_list.append(
                    soft_seq.augment_image(
                        current_img_list[
                            np.random.randint(current_n_img)
                        ]
                    )
                )
            h5_name = '%d.h5' % ord(label)
            with h5py.File('./processed_data/%s' % h5_name, 'w') as handle:
                handle.create_dataset('label', data=label)
                idx = 0
                for idx, img in enumerate(current_img_list):
                    handle.create_dataset(str(idx), data=img)
                for jdx, img in enumerate(new_img_list):
                    handle.create_dataset(str(idx+jdx+1), data=img)
            print(
                '[%s] Finished data augmentation (from %d to %d) for %s.' %
                (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    current_n_img,
                    max_n_img,
                    label
                )
            )
            return label, h5_name

        print(
            'Starting data augmentation at %s...' %
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        )

        h5_dict = dict()
        with pp.ProcessPool() as pool:
            arg_list = [
                do,
                list(self.total_result.keys()),
                list(self.total_result.values()),
                [max_n_img_per_char] * len(self.total_result)
            ]
            for char, file_name in pool.imap(*arg_list):
                h5_dict[char] = [file_name, max_n_img_per_char]

        with open('./processed_data/summary.json', 'w') as file:
            json.dump(h5_dict, file)

        print(
            'Finished data augmentation at %s...' %
            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        )

    def show_count_of_chars(self):
        plt.figure()
        plt.bar(
            self.total_result.keys(),
            [
                len(self.total_result[key])
                for key in self.total_result
            ]
        )
        plt.show()

    def show_n_squared_of_char(self, character, n, figsize=(16, 12)):
        img_idx_list = list(
            np.random.choice(
                len(self.total_result[character]),
                min([n ** 2, len(self.total_result[character])]),
                replace=False
            )
        )
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(nrows=n, ncols=n)
        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                img_idx = img_idx_list[
                    min(
                        [
                            col_idx + row_idx * n,
                            len(img_idx_list) - 1
                        ]
                    )
                ]
                img = self.total_result[character][img_idx]
                ax.imshow(img)
                ax.set_title(img_idx)
        fig.suptitle(character)
        fig.show()


class ImgSet(object):
    def __init__(self, path_to_img_set):
        """

        ds = ImgSet('./processed_data')
        ds.save_n_squared_samples_for_all_chars()

        """
        self.path_to_img_set = os.path.abspath(
            path_to_img_set
        )
        with open(os.path.join(self.path_to_img_set, 'summary.json')) as file_handle:
            self.summary = json.load(file_handle)
        self.char_to_h5_mapper = {
            key: h5py.File(
                os.path.join(
                    self.path_to_img_set,
                    self.summary[key][0]
                )
            )
            for key in self.summary
        }

    def show_n_squared_of_char(self, character, vertical_n_per_char, horizontal_n_per_char, figsize=(16, 12), show_img=True):
        img_idx_list = list(
            np.random.choice(
                len(self.char_to_h5_mapper[character]) - 1,
                min([vertical_n_per_char*horizontal_n_per_char, len(self.char_to_h5_mapper[character])]),
                replace=False
            )
        )
        fig = plt.figure(figsize=figsize)
        axes = fig.subplots(nrows=vertical_n_per_char, ncols=horizontal_n_per_char)
        for row_idx, row in enumerate(axes):
            for col_idx, ax in enumerate(row):
                img_idx = img_idx_list[
                    min(
                        [
                            col_idx + row_idx * horizontal_n_per_char,
                            len(img_idx_list) - 1
                        ]
                    )
                ]
                img = self.char_to_h5_mapper[character][str(img_idx)].value
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(str(img_idx))
        fig.suptitle(character)
        if show_img:
            fig.show()
        return fig

    def save_n_squared_samples_for_all_chars(self, vertical_n_per_char=5, horizontal_n_per_char=7, figsize=(16, 12)):
        if not os.path.exists('./samples_of_processed_data'):
            os.makedirs('./samples_of_processed_data')

        for character in tqdm(self.summary):
            fig = self.show_n_squared_of_char(
                character, vertical_n_per_char, horizontal_n_per_char, figsize=figsize, show_img=False
            )
            fig.savefig(
                './samples_of_processed_data/%s.png' %
                self.summary[character][0][:-3]
            )
            plt.close('all')
