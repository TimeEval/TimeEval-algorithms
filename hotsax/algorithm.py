#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from saxpy.hotsax import find_discords_hotsax
from typing import Optional


@dataclass
class CustomParameters:
    anomaly_window_size: int = 100
    paa_transform_size: int = 3
    alphabet_size: int = 3
    normalization_threshold: float = 0.01
    random_state: int = 42
    num_discords: Optional[int] = None


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


def main(args: AlgorithmArgs):
    set_random_state(args)
    data = np.genfromtxt(args.dataInput, skip_header=1, delimiter=",", usecols=(1,))

    window_size = args.customParameters.anomaly_window_size
    paa_size = args.customParameters.paa_transform_size
    if window_size < paa_size:
        print(f"anomaly_window_size ({window_size}) < paa_transform_size ({paa_size})! Therefore, we set paa_transform_size = anomaly_window_size.")
        paa_size = window_size
    num_discords = args.customParameters.num_discords
    if not num_discords:
        print(f"Searching for all discords")
        num_discords = len(data - window_size + 1)
    discords = find_discords_hotsax(data, num_discords=num_discords, sax_type='unidim',
        win_size=window_size,
        alphabet_size=args.customParameters.alphabet_size,
        paa_size=paa_size,
        znorm_threshold=args.customParameters.normalization_threshold
    )
    print(f"Found {len(discords)} discords")

    discord_idxs = [e[0] for e in discords]
    discord_scores = [e[1] for e in discords]
    df = pd.DataFrame(index=range(len(data)), dtype=np.float64)
    df["nn_distance"] = .0
    df.loc[discord_idxs, "nn_distance"] = discord_scores
    df.to_csv(args.dataOutput, index=False, header=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified, expected a single json-string!")
        exit(1)

    args = AlgorithmArgs.from_sys_args()
    print(f"Config: {args}")

    if args.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)

    main(args)
