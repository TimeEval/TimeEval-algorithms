#!/usr/bin/env python3
import json
import sys
import argparse
from dataclasses import dataclass, asdict

from typing import Optional
import numpy as np

from sarima import SARIMA


@dataclass
class CustomParameters:
    train_window_size: int = 500         # Number of points from the beginning of the series to build model on
    prediction_window_size: int = 10     # Number of points to forecast in one go; smaller = slower, but more accurate
    max_lag: Optional[int] = None        # Refit SARIMA model after that number of points (only helpful if fixed_orders=None)
    period: int = 1                      # >= 1 (if ==1: non-seasonal)
    max_iter: int = 50                   # smaller = faster, but might not converge
    exhaustive_search: bool = False      # performs full grid search to find optimal SARIMA-model --> SLOW!
    n_jobs: int = 1                      # only used for grid search
    fixed_orders: Optional[dict] = None  # allows specifying the orders; if set, skips the AutoARIMA-search
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


def load_data(config: AlgorithmArgs) -> np.ndarray:
    return np.genfromtxt(config.dataInput, delimiter=",", skip_header=True, usecols=(1,))


def main(config: AlgorithmArgs):
    set_random_state(config)
    data = load_data(config)

    params = asdict(config.customParameters)
    del params["random_state"]
    model = SARIMA(**params)
    scores = model.fit_predict(data)

    print(f"Writing results to {config.dataOutput}")
    np.savetxt(config.dataOutput, scores, delimiter=",")

    #plot(model, data)


def plot(model, data):
    import pandas as pd
    import matplotlib.pyplot as plt

    predictions = model._predicitions
    scores = model._scores

    plt.figure()
    df = pd.DataFrame({"data": data, "predictions": predictions, "scores": scores})
    df.plot()
    plt.legend()
    plt.show()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
