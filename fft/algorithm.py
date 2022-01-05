import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from fft import detect_anomalies


@dataclass
class CustomParameters:
    ifft_parameters: int = 5
    context_window_size: int = 21
    local_outlier_threshold: float = .6
    max_anomaly_window_size: int = 50
    max_sign_change_distance: int = 10
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.values[:, 1]

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

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


def execute(args: AlgorithmArgs) -> None:
    set_random_state(args)
    data = args.ts
    anomaly_scores = detect_anomalies(
        data,
        max_region_size=args.customParameters.max_anomaly_window_size,
        local_neighbor_window=args.customParameters.context_window_size,
        **asdict(args.customParameters)
    )
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)
    args = AlgorithmArgs.from_sys_args()
    print(f"Config: {args}")

    if args.executionType == "train":
        print("Nothing to train, finished!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
