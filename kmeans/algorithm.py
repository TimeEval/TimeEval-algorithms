import numpy as np
import pandas as pd
import json
import sys
from dataclasses import dataclass, asdict
import argparse

from kmeans.model import KMeansAD


@dataclass
class CustomParameters:
    n_clusters: int = 20
    anomaly_window_size: int = 20
    stride: int = 1
    n_jobs: int = 1
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.iloc[:, 1:-1].values

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


def execute(args: AlgorithmArgs):
    set_random_state(args)
    data = args.ts
    params = asdict(args.customParameters)
    params["k"] = params["n_clusters"]
    params["window_size"] = params["anomaly_window_size"]
    del params["n_clusters"]
    del params["random_state"]
    del params["anomaly_window_size"]
    detector = KMeansAD(**params)
    anomaly_scores = detector.fit_predict(data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
