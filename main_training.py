#!/usr/bin/env python

from chRecog.models import fcn
from chRecog.searchers import GridSearcher


params = {
    'n_epoch': 1000,
    'batch_size': 32,
    'lr': {0: 1e-3, 5: 1e-4, 16: 1e-5, 31: 1e-6},
    'l2': 1.0,
    'keep_prob': [0.7, 0.8, 0.9],
    'random_seed': 66,
    'optimizer': 'adam'
}

searcher_params = {
    'model': fcn,
    'device_str': '0',
    'keep_prob': 1.0
}


if __name__ == '__main__':
    gs = GridSearcher(params)
    gs.start(**searcher_params)
