#!/usr/bin/env python3

import json
import sys
import argparse
import numpy as np

from dataclasses import dataclass

from ptsa.models.arima import ARIMA
from ptsa.models.distance import Euclidean
from ptsa.models.distance import Mahalanobis
from ptsa.models.distance import Garch
from ptsa.models.distance import SSA
from ptsa.models.distance import Fourier
from ptsa.models.distance import DTW
from ptsa.models.distance import EDRS
from ptsa.models.distance import TWED


@dataclass
class CustomParameters:
    window_size: int = 20
    max_lag: int = 30000
    p_start: int = 1
    q_start: int = 1
    max_p: int = 5
    max_q: int = 5
    differencing_degree: int = 0
    distance_metric: str = "Euclidean"
    random_state: int = 42  # seed for randomness


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        if len(sys.argv) != 2:
            raise ValueError("Wrong number of arguments specified! Single JSON-string pos. argument expected.")
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def distance_to_measure(distance_metric):
    switcher = {
        "euclidean": Euclidean(),
        "mahalanobis": Mahalanobis(),
        "garch": Garch(),
        "ssa": SSA(),
        "fourier": Fourier(),
        "dtw": DTW(),
        "edrs": EDRS(),
        "twed": TWED()
    }
    return switcher.get(distance_metric.lower(), "missing")


def main():
    config = AlgorithmArgs.from_sys_args()
    ts_filename = config.dataInput  # "/data/dataset.csv"
    score_filename = config.dataOutput  # "/results/anomaly_window_scores.ts"

    print(f"Configuration: {config}")

    if config.executionType == "train":
        print("No training required!")
        exit(0)

    if config.executionType != "execute":
        raise ValueError("Unknown executionType specified!")

    set_random_state(config)

    # read only single "value" column from dataset
    print(f"Reading data from {ts_filename}")
    da = np.genfromtxt(ts_filename, skip_header=1, delimiter=",")
    data = da[:, 1]
    labels = da[:, -1]
    length = len(data)
    contamination = labels.sum() / length
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination

    # run ARIMA
    print("Executing ARIMA ...")
    model = ARIMA(
       window=config.customParameters.window_size,
       max_lag=config.customParameters.max_lag,
       p_start=config.customParameters.p_start,
       q_start=config.customParameters.q_start,
       max_p=config.customParameters.max_p,
       max_q=config.customParameters.max_q,
       d=config.customParameters.differencing_degree,
       contamination=contamination,
       neighborhood="all")
    model.fit(data)

    # get outlier scores
    measure = distance_to_measure(config.customParameters.distance_metric)
    if measure == "missing":
        raise ValueError(f"Distance measure '{config.customParameters.distance_metric}' not supported!")
    measure.detector = model
    measure.set_param()
    model.decision_function(measure=measure)
    scores = model.decision_scores_

    #from ptsa.utils.metrics import metricor
    #grader = metricor()
    #preds = grader.scale(scores, 0.1)

    print(f"Input size: {len(data)}\nOutput size: {len(scores)}")
    print("ARIMA result:", scores)

    print(f"Writing results to {score_filename}")
    np.savetxt(score_filename, scores, delimiter=",")


if __name__ == "__main__":
    main()
