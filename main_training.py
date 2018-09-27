#!/usr/bin/env python

from chrecog.models import fcn, afcn, bfcn
from chrecog.searchers import GridSearcher


params = {
    'n_epoch': 1000,
    'batch_size': 32,
    'lr': {0: 1e-3, 31: 1e-4, 61: 1e-5, 91: 3e-6, 121: 1e-6},
    # 'lr': {0: 1e-4, 31: 1e-5, 61: 1e-6},
    'l2': 1.0,
    'keep_prob': [0.6, 0.8],
    'random_seed': 66,
    'optimizer': 'adam'
}

searcher_params = {
    'model': bfcn,
    'device_str': '0',
    'keep_prob': 1.0
}


if __name__ == '__main__':

    gs = GridSearcher(params)
    gs.start(**searcher_params)
