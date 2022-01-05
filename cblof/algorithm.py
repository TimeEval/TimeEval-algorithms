#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from pyod.models.cblof import CBLOF


@dataclass
class CustomParameters:
    n_clusters: int = 8
    alpha: float = 0.9
    beta: float = 5
    use_weights: bool = False
    random_state: int = 42
    n_jobs: int = 1


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    contamination = labels.sum() / len(labels)
    # Use smallest positive float as contamination if there are no anomalies in dataset
    contamination = np.nextafter(0, 1) if contamination == 0. else contamination
    return data, contamination


def main(config: AlgorithmArgs):
    data, contamination = load_data(config)

    clf = CBLOF(
        contamination=contamination,
        n_clusters=config.customParameters.n_clusters,
        alpha=config.customParameters.alpha,
        beta=config.customParameters.beta,
        use_weights=config.customParameters.use_weights,
        random_state=config.customParameters.random_state,
        n_jobs=config.customParameters.n_jobs,
        check_estimator=False,
        clustering_estimator=None,
    )
    clf.fit(data)
    scores = clf.decision_scores_
    np.savetxt(config.dataOutput, scores, delimiter=",")


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
