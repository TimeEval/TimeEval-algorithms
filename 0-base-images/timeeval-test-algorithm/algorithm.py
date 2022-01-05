import numpy as np
import pandas as pd
import json
import time
import argparse
import sys


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        dataset = pd.read_csv(self.dataInput)
        return dataset.values[:, 1:-1]

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args = json.loads(sys.argv[1])
        return AlgorithmArgs(**args)


def main():
    args = AlgorithmArgs.from_sys_args()
    will_raise = args.customParameters.get("raise", False)
    sleep_seconds = args.customParameters.get("sleep", 10)

    if will_raise:
        raise Exception("from within")

    time.sleep(sleep_seconds)

    np.zeros(args.ts.shape[0]).tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    main()
