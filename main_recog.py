#!/usr/bin/env python

import os
import cv2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from chrecog.loaders import AugOCRSet
from chrecog.engine import CharRecogEngine
from preparation.imgmaker import DataSet


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

number = 10
# path = 'D:\projects\ch_ocr\good_logs\log_2018_09_24_16_26_36_510687\saved_models\epoch_195'
path = 'D:\projects\ch_ocr\good_logs\log_2018_09_26_11_24_05_455420\saved_models\epoch_155'
# path = 'D:\projects\ch_ocr\logs\log_2018_09_27_18_15_27_489283\saved_models\epoch_120'

loader = AugOCRSet()
engine = CharRecogEngine(path, loader.label_to_char_mapper)

rand_batch, batch_label = loader.get_random_samples(number)
pred_batch, batch_lik = engine.predict(rand_batch)

print(pred_batch.flatten())
print(batch_label)


ds = DataSet()
ds.get_sub_folder(0)
ds.get_an_orgbox_image_and_csv(0)

img, _, truth = ds.get_a_box_and_title(7)
plt.imshow(img)

pred, lik = engine.scan(img)

print([truth, pred[lik > lik.mean()]])