import argparse
from dataclasses import dataclass
from typing import Tuple
import sys
import json

import numpy as np

from tsbitmapper import TSBitMapper


@dataclass
class CustomParameters:
    feature_window_size: int = 100
    alphabet_size: int = 5
    level_size: int = 3
    lead_window_size: int = 200
    lag_window_size: int = 300
    compression_ratio: int = 2
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
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


def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    return np.genfromtxt(path,
                         skip_header=1,
                         delimiter=",",
                         usecols=[1])


def execute(config):
    set_random_state(config)
    anom_timeseries_1d = load_data(config.dataInput)

    BMP = TSBitMapper(feature_window_size=config.customParameters.feature_window_size,
                      bins=config.customParameters.alphabet_size,
                      level_size=config.customParameters.level_size,
                      lead_window_size=config.customParameters.lead_window_size,
                      lag_window_size=config.customParameters.lag_window_size,
                      compression_ratio=config.customParameters.compression_ratio)

    scores = BMP.fit_predict(np.squeeze(anom_timeseries_1d))
    decompressed_scores = BMP.post_ts_bitmap(scores)
    np.savetxt(config.dataOutput, decompressed_scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()

    if config.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'execute'!")
