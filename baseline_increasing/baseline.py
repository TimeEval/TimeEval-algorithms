import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
import sys


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts_length(self) -> int:
        return self.df.shape[0]

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        return AlgorithmArgs(**args)


def execute(args: AlgorithmArgs):
    indices = np.arange(args.ts_length)
    anomaly_scores = MinMaxScaler().fit_transform(indices.reshape(-1, 1)).reshape(-1)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
