#!/usr/bin/env python

from chRecog.models import all_conv_nn
from chRecog.searchers import GridSearcher


params = {
    'n_epoch': 1000,
    'batch_size': 64,
    'lr': 1e-4,
    'l2': [1e-6, 1e-7],
    'random_seed': 66,
    'optimizer': 'adam',
    'keep_prob': 1.0    # ignore
}

searcher_params = {
    'model': all_conv_nn,
    'device_str': '0',
    'keep_prob': 1.0
}


if __name__ == '__main__':
    gs = GridSearcher(params)
    gs.start(**searcher_params)
