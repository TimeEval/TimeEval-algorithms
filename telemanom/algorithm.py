from telemanom.detector import Detector
from telemanom.modeling import Model
from telemanom.helpers import Config
from telemanom.channel import Channel
import argparse
import pandas as pd
import numpy as np
import json
import sys
from dataclasses import dataclass, asdict, field
from typing import List
from tensorflow.compat.v1 import set_random_seed


CHANNEL_ID = "0"


@dataclass
class CustomParameters:
    batch_size: int = 70
    smoothing_window_size: int = 30
    smoothing_perc: float = 0.05
    error_buffer: int = 100
    loss_metric: str = 'mse'
    optimizer: str = 'adam'
    split: float = 0.8
    dropout: float = 0.3
    lstm_batch_size: int = 64
    epochs: int = 35
    layers: List[int] = field(default_factory=lambda: [80, 80])
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.0003
    window_size: int = 250
    prediction_window_size: int = 10
    p: float = 0.13
    use_id: str = "internal-run-id"
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        dataset = pd.read_csv(self.dataInput)
        return dataset.values[:, 1:-1]

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def adapt_config_yaml(args: AlgorithmArgs) -> Config:
    params = asdict(args.customParameters)
    # remap config keys
    params["validation_split"] = 1 - params["split"]
    params["patience"] = params["early_stopping_patience"]
    params["min_delta"] = params["early_stopping_delta"]
    params["l_s"] = params["window_size"]
    for k in ["split", "early_stopping_patience", "early_stopping_delta"]:
        del params[k]

    config = Config.from_dict(params)
    if args.executionType == "train":
        config["train"] = True
        config["predict"] = False
    elif args.executionType == "execute":
        config["train"] = False
        config["predict"] = True

    return config


def train(args: AlgorithmArgs, config: Config, channel: Channel):
    Model(config, config.use_id, channel, model_path=args.modelOutput)  # trains and saves model


def execute(args: AlgorithmArgs, config: Config, channel: Channel):
    detector = Detector(config=config, model_path=args.modelInput, result_path=args.dataOutput)
    errors = detector.predict([channel])[0]
    errors.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    set_random_seed(seed)


def main():
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    config = adapt_config_yaml(args)
    is_train = args.executionType == "train"

    single_channel = Channel(config, CHANNEL_ID)
    single_channel.set_data(args.ts, train=is_train)

    if is_train:
        train(args, config, single_channel)
    else:
        execute(args, config, single_channel)


if __name__ == "__main__":
    main()
