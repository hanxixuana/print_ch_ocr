#!/usr/bin/env python

from chRecog.searchers import GridSearcher


params = {
    'n_epoch': 1000,
    'batch_size': [16, 32],
    'lr': 1e-5,
    'l2': [1e-1, 5e-1, 1e-2],
    'keep_prob': 0.5,
    'random_seed': 666,
    'optimizer': 'adam'
}

searcher_params = {
    'device_str': '0',
    'keep_prob': 1.0
}


if __name__ == '__main__':

    gs = GridSearcher(params)
    gs.start(**searcher_params)
