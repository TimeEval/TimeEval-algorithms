#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import pandas as pd

from phasespace import detect_anomalies


@dataclass
class CustomParameters:
    embed_dim_range: List[int] = field(default_factory=lambda: [50, 100, 150])
    project_phasespace: bool = False
    nu: float = 0.5
    gamma: Optional[float] = None
    kernel: str = "rbf"
    degree: int = 3
    coef0: float = 0.0
    tol: float = 0.001
    random_state: int = 42
    shrinking: bool = True  # using default is fine
    cache_size: float = 200  # using default is fine
    max_iter: int = -1  # using default is fine


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    data = df.iloc[:, 1:-1].values
    return data


def main(config: AlgorithmArgs):
    set_random_state(config)
    data = load_data(config)
    scores = detect_anomalies(data.ravel(),
                              embed_dims=config.customParameters.embed_dim_range,
                              projected_ps=config.customParameters.project_phasespace,
                              nu=config.customParameters.nu,
                              gamma=config.customParameters.gamma or "scale",
                              kernel=config.customParameters.kernel,
                              degree=config.customParameters.degree,
                              coef0=config.customParameters.coef0,
                              tol=config.customParameters.tol,
                              shrinking=config.customParameters.shrinking,
                              cache_size=config.customParameters.cache_size,
                              max_iter=config.customParameters.max_iter,
                              )
    np.savetxt(config.dataOutput, scores, delimiter=",")


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
