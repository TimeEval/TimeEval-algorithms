#!/usr/bin/env python3
import json
import sys
from argparse import Namespace
from dataclasses import dataclass

import numpy as np

from multinormats import MultiNormA


@dataclass
class CustomParameters:
    anomaly_window_size: int = 20
    normal_model_percentage: float = 0.5
    max_motifs: int = 4096
    random_state: int = 42
    motif_detection: str = "mixed"
    sum_dims: bool = False
    normalize_join: bool = True
    join_combine_method: int = 1


class AlgorithmArgs(Namespace):
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


def main():
    config = AlgorithmArgs.from_sys_args()
    ts_filename = config.dataInput  # "/data/dataset.csv"
    score_filename = config.dataOutput  # "/results/anomaly_window_scores.ts"
    execution_type = config.executionType
    # we ignore model paths, because they are not required
    window_size = config.customParameters.anomaly_window_size
    normal_model_percentage = config.customParameters.normal_model_percentage
    max_motifs = config.customParameters.max_motifs
    motif_detection = config.customParameters.motif_detection
    if motif_detection not in ["stomp", "random", "mixed"]:
        raise ValueError(f"motif_detection (={motif_detection}) must be one of [stomp,random,mixed]!")
    sum_dims = config.customParameters.sum_dims
    normalize_join = config.customParameters.normalize_join
    join_combine_method = config.customParameters.join_combine_method

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
    with open(ts_filename, 'r') as f:
        num_cols = len(f.readline().split(","))
        f.close()

    data = np.genfromtxt(ts_filename, skip_header=1, delimiter=",", usecols=range(1, num_cols - 1))
    length = len(data)

    # save as a new file to pass to NormA
    ts_transformed_name = f"transformed.csv"
    np.savetxt(ts_transformed_name, data, delimiter=",")

    # window_size = window_size + np.random.randint(0, 3 * window_size)
    window_size = max(10, window_size)

    # Run NormA
    print("Executing MultiNormA ...")
    norma = MultiNormA(pattern_length=window_size, nm_size=3 * window_size, motif_detection=motif_detection,
                       sum_dims=sum_dims, apply_normalize_join=normalize_join,
                       combine_method=join_combine_method)
    scores = norma.run_motif(ts_transformed_name, tot_length=length, percentage_sel=normal_model_percentage,
                             max_motifs=max_motifs)
    print(f"Input size: {len(data)}\nOutput size: {len(scores)}")
    print("MultiNormA (random NM) result:", scores)

    print(f"Writing results to {score_filename}")
    np.savetxt(score_filename, scores, delimiter=",")


if __name__ == "__main__":
    main()
