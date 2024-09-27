#!/usr/bin/env python3
import json
import sys
from argparse import Namespace

import numpy as np

from dataclasses import dataclass
from normats import NormA


@dataclass
class CustomParameters:
    anomaly_window_size: int = 20
    normal_model_percentage: float = 0.5
    random_state: int = 42


class AlgorithmArgs(Namespace):
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


def main():
    config = AlgorithmArgs.from_sys_args()
    ts_filename = config.dataInput  # "/data/dataset.csv"
    score_filename = config.dataOutput  # "/results/anomaly_window_scores.ts"
    execution_type = config.executionType
    # we ignore model paths, because they are not required
    window_size = config.customParameters.anomaly_window_size
    normal_model_percentage = config.customParameters.normal_model_percentage
    # postprocessing window_size = 2 * (window_size - 1) + 1

    set_random_state(config)

    print(f"Configuration: {config}")

    if execution_type == "train":
        print("No training required!")
        exit(0)
    elif execution_type != "execute":
        raise ValueError(f"Unknown execution type '{execution_type}'; expected either 'train' or 'execute'!")

    # read only single "value" column from dataset
    print(f"Reading data from {ts_filename}")
    data = np.genfromtxt(ts_filename, skip_header=1, delimiter=",", usecols=(1,))
    length = len(data)

    # save as a new file to pass to NormA
    ts_transformed_name = f"transformed.csv"
    np.savetxt(ts_transformed_name, data, delimiter=",")

    # Run NomrA
    print("Executing NormA ...")
    norma = NormA(pattern_length=window_size, nm_size=3 * window_size)
    scores = norma.run_motif(ts_transformed_name, tot_length=length, percentage_sel=normal_model_percentage)
    print(f"Input size: {len(data)}\nOutput size: {len(scores)}")
    print("NormA (random NM) result:", scores)

    print(f"Writing results to {score_filename}")
    np.savetxt(score_filename, scores, delimiter=",")


if __name__ == "__main__":
    main()
