#!/usr/bin/env python3

import sys
import json
import math
import heapq

import numpy as np
from typing import Optional
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest


class Config:
    dataInput: Path
    dataOutput: Path
    executionType: str
    n_trees: int
    max_samples: Optional[float]
    n_neighbors: int
    alpha: float
    m: int
    random_state: int

    def __init__(self, params):
        self.dataInput = Path(params.get('dataInput',
                                         '/data/dataset.csv'))
        self.dataOutput = Path(params.get('dataOutput',
                                          '/results/anomaly_window_scores.ts'))
        self.executionType = params.get('executionType',
                                        'execute')
        try:
            customParameters = params['customParameters']
        except KeyError:
            customParameters = {}
        self.n_trees = customParameters.get('n_trees', 200)
        self.max_samples = customParameters.get('max_samples', None)
        self.n_neighbors = customParameters.get('n_neighbors', 20)
        self.alpha = customParameters.get('alpha', 1)
        self.m = customParameters.get('m', None)
        self.random_state = customParameters.get('random_state', 42)


def set_random_state(config) -> None:
    seed = config.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def read_data(config: Config):
    print('Reading Data...')
    X = np.genfromtxt(config.dataInput, delimiter=',', skip_header=True)
    # skip first col (index) and last col (label)
    X = X[:, 1:-1]
    print('Data')
    print('  dims:', X.shape[0])
    print('  samples:', X.shape[1])
    if config.m is None:
        config.m = len(X[0])
    return (X, config)


def compute_iforest_scores(X, config: Config):
    print('Configuring forest...')
    max_samples = config.max_samples if config.max_samples else "auto"
    forest = IsolationForest(n_estimators=config.n_trees,
                             max_samples=max_samples,
                             random_state=config.random_state)
    forest.fit(X)
    print('Computing Forest scores...')
    scores = forest.decision_function(X)
    return -scores


def save_results(data, path: str):
    print(f'Saving Results to {path}')
    np.savetxt(path, data, delimiter=',', fmt='%f')
    print('Results saved')


def outlier_coefficient_for_attribute(attr_index: int, data):
    ''' The original paper is incorrect and inaccurate over here.
    My assumption is that we would want to calculate the following:
    | emperical_standard_deviation(attr) / mean(attr) | '''

    attr = data[:, attr_index]
    mean = np.mean(attr)
    esd = np.std(attr)
    # We take to absolute value in the case of a negative mean
    return np.abs(esd / mean)


def prune_data(config: Config, data, anomaly_scores):
    ''' The original paper is very inaccurate over here and it is sometimes hard
    to grasp the meaning of variables. Please be aware that
    this method might not be the same as inteded by the authors, but is my
    assumption on what they were trying to do. The pruning is described
    in section 3.3 of the paper'''

    print('Pruning data...')

    outlier_coefficients = [outlier_coefficient_for_attribute(attr_index, data)
                            for attr_index in range(len(data[0]))]

    # assumption: We want to get the m outlier coefficients with highest value
    outlier_coefficients.sort(reverse=True)
    top_m = outlier_coefficients[0:config.m]
    proportion_of_outliers = (config.alpha * sum(top_m)) / config.m

    # now that we know the proportion of outliers, we return the according
    # amount of data points with the highest anomaly scores
    num_outliers = math.ceil(len(data) * proportion_of_outliers)
    print(f'Num of outlier_candidates {num_outliers}')

    # prune the dataset by removing all points except the num_outlier points with
    # highest anomaly score
    min_anomaly_score = heapq.nlargest(num_outliers, anomaly_scores)[-1]
    outlier_candidates_indexes = [i for i in range(len(data))
                                  if anomaly_scores[i] > min_anomaly_score]
    outlier_candidates = [data[i] for i in outlier_candidates_indexes]

    return (outlier_candidates, outlier_candidates_indexes)


def compute_lof(config: Config, data, outlier_canidates):
    print('Computing local outlier factors ...')
    lof = LocalOutlierFactor(n_neighbors=config.n_neighbors, novelty=True)
    lof.fit(data)
    return -lof.score_samples(outlier_canidates)


def continous_scores(outlier_factors, outlier_indexes, original_ds_len):
    print("Postprocessing")
    current_outlier_index = 0
    res = []

    def is_index_of_outlier_candidate(i):
        return i in outlier_indexes

    for i in range(0, original_ds_len):
        if is_index_of_outlier_candidate(i):
            res.append(outlier_factors[current_outlier_index])
            current_outlier_index += 1
        else:
            res.append(0)

    return res


def execute(config: Config):
    data, config = read_data(config=config)
    iforest_scores = compute_iforest_scores(X=data, config=config)
    outlier_candidates, outlier_indexes = prune_data(config=config,
                                                     data=data,
                                                     anomaly_scores=iforest_scores)
    outlier_factors = compute_lof(config=config,
                                  data=data,
                                  outlier_canidates=outlier_candidates)

    results = continous_scores(outlier_factors=outlier_factors,
                               outlier_indexes=outlier_indexes,
                               original_ds_len=len(data))

    save_results(data=results, path=config.dataOutput)


def parse_args():
    print(sys.argv)
    if len(sys.argv) < 2:
        print('No arguments supplied, using default arguments!',
              file=sys.stderr)
        params = {}
    elif len(sys.argv) > 2:
        print('Wrong number of arguments given! Single JSON-String expected!',
              file=sys.stderr)
        exit(1)
    else:
        params = json.loads(sys.argv[1])
    return Config(params)


if __name__ == '__main__':
    config = parse_args()
    if config.executionType == 'train':
        print('Nothing to train.')
    elif config.executionType == 'execute':
        execute(config)
    else:
        raise Exception('Invalid Execution type given')
