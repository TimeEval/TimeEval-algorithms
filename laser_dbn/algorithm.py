import argparse
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
import sys

from laser_dbn.model import DynamicBayesianNetworkAD


@dataclass
class CustomParameters:
    timesteps: int = 2
    n_bins: int = 10
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


def train(args: AlgorithmArgs):
    data = args.ts
    model = DynamicBayesianNetworkAD(timesteps=args.customParameters.timesteps,
                                     discretizer_n_bins=args.customParameters.n_bins)
    model.fit(data)
    model.save(args.modelOutput)


def execute(args: AlgorithmArgs):
    data = args.ts
    model = DynamicBayesianNetworkAD.load(args.modelInput)
    scores = model.predict(data)
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)



if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
