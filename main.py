#!/usr/bin/env python

import os
import numpy as np

from chRecog.loaders import AugOCRSet
from chRecog.models import deepnn
from chRecog.trainers import Trainer


params = {
    'n_epoch': 1000,
    'batch_size': 64,
    'lr': 1e-4,
    'l2': 1e-2,
    'keep_prob': 0.6,
    'random_seed': 666
}


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    np.random.seed(params['random_seed'])

    loader = AugOCRSet()
    loader.split_data_into_tvt((0.7, 0.3, 0.0))

    trainer = Trainer(deepnn, loader)
    trainer.train(params)
