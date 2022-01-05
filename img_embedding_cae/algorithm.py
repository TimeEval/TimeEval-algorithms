import argparse
from dataclasses import dataclass, asdict
import sys
import json

import numpy as np
from torch import nn

from img_embedding_cae.cae import CAE


downscaling_factor = 8
param_correction = True


@dataclass
class CustomParameters:
    anomaly_window_size: int = 512
    kernel_size: int = 2
    num_kernels: int = 64
    latent_size: int = 100
    leaky_relu_alpha: float = 0.03
    batch_size: int = 32
    test_batch_size: int = 128
    learning_rate: float = 0.001
    epochs: int = 30
    split: float = 0.8
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(path: str) -> np.ndarray:
    return np.genfromtxt(path,
                         skip_header=1,
                         delimiter=",",
                         usecols=[1])


def train(config: AlgorithmArgs):
    healthy_timeseries_1d = load_data(config.dataInput)
    parameters = asdict(config.customParameters)
    del parameters["random_state"]
    model = CAE(param_correction=param_correction, 
                downscaling_factor=downscaling_factor, 
                **parameters)
    model.fit(healthy_timeseries_1d, config.modelOutput)
    model.save(config.modelOutput)


def execute(config: AlgorithmArgs):
    anom_timeseries_1d = load_data(config.dataInput)
    model = CAE.load(config.modelInput)
    window_scores = model.predict_ts(anom_timeseries_1d, nn.L1Loss(reduction="sum"))
    # transform window scores to point scores and save
    window_scores = np.repeat(window_scores, model.anomaly_window_size)
    scores = np.zeros_like(anom_timeseries_1d)
    scores[:window_scores.shape[0]] = window_scores
    scores.tofile(config.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    set_random_state(config)

    if config.executionType == "train":
        train(config)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'train' or 'execute'!")
