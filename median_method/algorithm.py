import argparse
from dataclasses import dataclass
from typing import Tuple
import sys
import json

import numpy as np

from median_method import MedianMethod


@dataclass
class CustomParameters:
    neighbourhood_size: int = 100
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
    mm = MedianMethod(timeseries=anom_timeseries_1d,
                      neighbourhood_size=config.customParameters.neighbourhood_size)

    scores = mm.fit_predict()
    np.savetxt(config.dataOutput, scores)


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
