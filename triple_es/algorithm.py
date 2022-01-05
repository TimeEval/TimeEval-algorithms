import argparse
from dataclasses import dataclass, asdict
import sys
import json
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd

from triple_es import TripleES


@dataclass
class CustomParameters:
    train_window_size: int = 200
    period: int = 100
    trend: str = "add"
    seasonality: str = "add"
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


def load_data(path: str) -> np.ndarray:
    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True).iloc[:, 1].values


def execute(config):
    ts: np.ndarray = load_data(config.dataInput).reshape(-1, 1)
    ts = MinMaxScaler(feature_range=(0.1, 1.1)).fit_transform(ts).reshape(-1)  # data must be > 0
    parameters = asdict(config.customParameters)
    del parameters["random_state"]
    triple_es: TripleES = TripleES(ts, **parameters)
    scores = triple_es.fit_predict(plot=False)
    np.savetxt(config.dataOutput, scores)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)

    if config.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'execute'!")
