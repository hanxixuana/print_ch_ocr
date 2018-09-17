#!/usr/bin/env python

from multiprocessing import freeze_support

from preparation.imgmaker import DataSet

if __name__ == '__main__':

    freeze_support()

    ds = DataSet()
    result = ds.split_all_sub_folders(max_height=90, max_width=90)


