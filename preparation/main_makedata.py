#!/usr/bin/env python

from multiprocessing import freeze_support

from preparation.imgmaker import WordDataSet, WordImgSet, CharDataSet, CharImgSet

if __name__ == '__main__':

    freeze_support()

    ds = WordDataSet()
    ds.split_all_sub_folders(max_height=64)

    ds = WordImgSet('./processed_data/words.h5')
    ds.save_n_pics_of_samples()

    # ds = CharDataSet()
    # result = ds.split_all_sub_folders(max_height=64, max_width=32)
    #
    # ds = CharImgSet('./processed_data')
    # ds.save_n_squared_samples_for_all_chars()