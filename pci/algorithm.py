import numpy as np
import pandas as pd
import json
import sys
from dataclasses import dataclass
import argparse

from pci.model import PCIAnomalyDetector


@dataclass
class CustomParameters:
    window_size: int = 20
    thresholding_p: float = 0.05
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


def execute(args: AlgorithmArgs):
    set_random_state(args)
    data = args.ts
    pci = PCIAnomalyDetector(
        k=args.customParameters.window_size // 2,
        p=args.customParameters.thresholding_p,
        calculate_labels=False
    )
    anomaly_scores, _ = pci.detect(data)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
