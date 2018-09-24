#!/usr/bin/env python

import os
import time
import numpy as np

from sklearn.model_selection import ParameterGrid
from pathos import multiprocessing as multiprocessing

from chRecog.loaders import AugOCRSet
from chRecog.trainers import Trainer


class GridSearcher(object):
    def __init__(self, param_candidates):
        """

        class Fool(): pass
        self = Fool()

        param_candidates = {
            'n_epoch': 1000,
            'batch_size': [16, 32],
            'lr': 1e-4,
            'l2': [1e-1, 1e-2, 1e-3],
            'keep_prob': [0.5, 0.75],
            'random_seed': 666
        }

        """
        self.param_candidates = param_candidates
        self.all_params = list()
        self.param_to_result_mapper = dict()

    def assemble_params(self, keep_prob):
        single_candidate_params = {
            key: self.param_candidates[key]
            for key in self.param_candidates
            if not (
                isinstance(self.param_candidates[key], list)
                or
                isinstance(self.param_candidates[key], tuple)
            )
        }
        multiple_candidate_params = {
            key: self.param_candidates[key]
            for key in self.param_candidates
            if (
                isinstance(self.param_candidates[key], list)
                or
                isinstance(self.param_candidates[key], tuple)
            )
        }
        multiple_candidate_param_list = list(ParameterGrid(multiple_candidate_params))
        for item in multiple_candidate_param_list:
            if np.random.uniform(0.0, 1.0) <= keep_prob:
                item.update(single_candidate_params)
                self.all_params.append(item)

    @staticmethod
    def start_one_set(model, params, device_str):
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str
        np.random.seed(params['random_seed'])
        loader = AugOCRSet()
        loader.split_data_into_tvt((0.8, 0.2, 0.0))
        trainer = Trainer(loader)
        trainer.train(model, params)

    def start(self, model, device_str, keep_prob=1.0):
        self.assemble_params(keep_prob)

        pool = multiprocessing.Pool(
            processes=multiprocessing.cpu_count() - 1,
            maxtasksperchild=1
        )

        for params in self.all_params:
            time.sleep(2.0)
            pool.apply_async(self.start_one_set, args=(model, params, device_str))

        pool.close()
        pool.join()
