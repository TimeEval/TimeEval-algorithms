import pandas as pd
import json
import sys
from dataclasses import dataclass
import argparse

from msanomalydetector import SpectralResidual, DetectMode


@dataclass
class CustomParameters:
    mag_window_size: int = 3
    score_window_size: int = 40
    window_size: int = 50
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
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


def execute(args: AlgorithmArgs):
    series = args.df.iloc[:, [0, 1]]
    series.columns = ["timestamp", "value"]
    sr = SpectralResidual(series,
                          mag_window=args.customParameters.mag_window_size,
                          score_window=args.customParameters.score_window_size,
                          batch_size=args.customParameters.window_size,
                          sensitivity=99,
                          detect_mode=DetectMode.anomaly_only,
                          threshold=0.3)
    scores = sr.detect().score.values
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
