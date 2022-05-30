#!/usr/bin/env python3
import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dwt_mlead import DWT_MLEAD


@dataclass
class CustomParameters:
    start_level: int = 3
    quantile_epsilon: float = 0.01
    random_state: int = 42
    use_column_index: int = 0


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    column_index = 0
    if config.customParameters.use_column_index is not None:
        column_index = config.customParameters.use_column_index
    max_column_index = df.shape[1] - 3
    if column_index > max_column_index:
        print(f"Selected column index {column_index} is out of bounds (columns = {df.columns.values}; "
              f"max index = {max_column_index} [column '{df.columns[max_column_index + 1]}'])! "
              "Using last channel!", file=sys.stderr)
        column_index = max_column_index
    # jump over index column (timestamp)
    column_index += 1

    data = df.iloc[:, column_index].values
    return data


def main(config: AlgorithmArgs):
    np.random.seed(config.customParameters.random_state)

    data = load_data(config)
    detector = DWT_MLEAD(data,
                         start_level=config.customParameters.start_level,
                         quantile_boundary_type="percentile",
                         quantile_epsilon=config.customParameters.quantile_epsilon,
                         track_coefs=True,  # just used for plotting
                         )
    point_scores = detector.detect()

    # print("\n=== Cluster anomalies ===")
    # clusters = detector.find_cluster_anomalies(point_scores, d_max=2.5, anomaly_counter_threshold=2)
    # for c in clusters:
    #     print(f"  Anomaly at {c.center} with score {c.score}")

    # save individual point scores instead of cluster centers
    np.savetxt(config.dataOutput, point_scores, delimiter=",")
    print("\n=== Storing results ===")
    print(f"Saved **point scores** to {config.dataOutput}.")

    # detector.plot(coefs=False, point_anomaly_scores=point_scores)


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
