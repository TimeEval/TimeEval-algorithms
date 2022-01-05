#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Optional
import numpy as np
import pandas as pd

from dataclasses import asdict, dataclass
from model import NoveltySVR


@dataclass
class CustomParameters:
    n_init_train: int = 500  # initial data used to fit the regression model
    # removes training samples from the model that are older than forgetting_time
    forgetting_time: Optional[int] = None
    train_window_size: int = 16  # D = embedding dimension
    anomaly_window_size: int = 6  # n = event_duration (not too large)
    # confidence_level: float = 0.95  # c \in (0, 1)
    lower_suprise_bound: Optional[int] = None  # h = anomaly_window_size / 2
    scaling: str = "standard"  # one of "standard", "robust", "power" or empty/None
    epsilon: float = 0.1  # reused for SVR
    verbose: int = 0  # reused for SVR
    C: float = 1.0
    kernel: str = "rbf"
    degree: int = 3
    gamma: Optional[float] = None
    coef0: float = 0.0
    tol: float = 1e-3
    stabilized: bool = True
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> "AlgorithmArgs":
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(
                lambda x: x[0] in custom_parameter_keys,
                args.get("customParameters", {}).items(),
            )
        )
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    return df.iloc[:, 1].values.reshape(-1, 1)


def execute(config: AlgorithmArgs):
    set_random_state(config)
    plot = False

    data = load_data(config)
    train_split_at = config.customParameters.n_init_train

    params = asdict(config.customParameters)
    del params["n_init_train"]
    del params["random_state"]
    dect = NoveltySVR(**params)

    print("\n## TRAINING")
    dect.fit(data[:train_split_at])

    print("\n## DETECTING")
    scores = np.full(data.shape[0], fill_value=np.nan)
    scores[train_split_at:] = dect.detect(
        data[train_split_at:], plot=plot, train_skip=train_split_at
    )

    print("\n## WRITING OUTPUT")
    np.savetxt(config.dataOutput, scores, delimiter=",")
    if plot:
        import matplotlib.pyplot as plt

        train = dect.scaler.transform(data[:train_split_at]).reshape(-1)
        plt.plot(train, color="blue")
        plt.vlines(x=[train_split_at], ymin=train.min(), ymax=train.max(), color="red")
        plt.legend()
        plt.xlabel("Index / Time")
        # plt.savefig("fig.pdf")
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(f"Config: {config}")

    if config.executionType == "train":
        print("Nothing to train, finished!")
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(
            f"Unknown execution type '{config.executionType}'; "
            "expected either 'train' or 'execute'!"
        )
