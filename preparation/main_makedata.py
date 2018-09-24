#!/usr/bin/env python

from multiprocessing import freeze_support

from preparation.imgmaker import DataSet, ImgSet

if __name__ == '__main__':

    freeze_support()

    ds = DataSet()
    result = ds.split_all_sub_folders(max_height=64, max_width=32)

    ds = ImgSet('./processed_data')
    ds.save_n_squared_samples_for_all_chars()
