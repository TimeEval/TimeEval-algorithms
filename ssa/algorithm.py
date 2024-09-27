#!/usr/bin/env python3

import json
import sys
import argparse
import numpy as np

from dataclasses import dataclass
from typing import Union

from ptsa.models.SSA import SSA


@dataclass
class CustomParameters:
    ep: int = 3
    window_size: int = 720
    rf_method: str = 'alpha'
    alpha: Union[float, int, np.ndarray] = 0.2
    random_state: int = 42


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


def main():
    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)
    ts_filename = config.dataInput  # "/data/dataset.csv"
    score_filename = config.dataOutput  # "/results/anomaly_window_scores.ts"

    print(f"Configuration: {config}")

    if config.executionType == "train":
        print("No training required!")
        exit(0)

    if config.executionType != "execute":
        raise ValueError("Unknown executionType specified!")

    # read only single column from dataset
    print(f"Reading data from {ts_filename}")
    da = np.genfromtxt(ts_filename, skip_header=1, delimiter=",")
    data = da[:, 1]

    # run SSA
    print("Executing SSA ...")
    model = SSA(a=config.customParameters.alpha,
                ep=config.customParameters.ep,
                n=config.customParameters.window_size,
                rf_method=config.customParameters.rf_method)
    model.fit(data)

    # get outlier scores
    scores = model.decision_scores_
    scores = np.roll(scores, -config.customParameters.window_size)

    print(f"Input size: {len(data)}\nOutput size: {len(scores)}")
    print("SSA result:", scores)

    print(f"Writing results to {score_filename}")
    np.savetxt(score_filename, scores, delimiter=",", fmt='%f')


if __name__ == "__main__":
    main()
