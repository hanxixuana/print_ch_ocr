#!/usr/bin/env python

from chrecog.ctcmodels import CTC, DecoderType
from chrecog.searchers import GridSearcher


params = {
    'n_epoch': 300,
    'batch_size': 16,
    'lr': {0: 1e-4, 81: 1e-5, 151: 1e-6},
    'l2': 1.0,
    'keep_prob': 0.9,
    'max_text_len': 32,
    'random_seed': 888,
    'decoder_type': DecoderType.BestPath,
    'optimizer': 'rmsprop',
    'verbose': True
}

searcher_params = {
    'model': CTC,
    'device_str': '0',
    'keep_prob': 1.0
}


if __name__ == '__main__':

    gs = GridSearcher(params)
    gs.start(**searcher_params)
