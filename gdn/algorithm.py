#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch

from dataclasses import dataclass

from GDN.main import GDNtrain, GDNtest


@dataclass
class CustomParameters:
    window_size: int = 15
    stride: int = 5
    latent_size: int = 64
    n_out_layers: int = 1
    out_layer_dimensionality: int = 1
    epochs: int = 1
    batch_size: int = 128
    split: float = 0.9
    learning_rate_decay: float = 0.001
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> pd.DataFrame:
        return self.df.iloc[:, 1:-1]

    @property
    def tsa(self) -> pd.Dataframe:
        return self.df.iloc[:, 1:]

    @property
    def df(self) -> pd.DataFrame:
        return pd.read_csv(self.dataInput)

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(
            filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def train(args: AlgorithmArgs):
    ts = args.ts

    train_config = {
        "batch": args.customParameters.batch_size,
        "epoch": args.customParameters.epochs,
        "slide_win": args.customParameters.window_size,
        "dim": args.customParameters.latent_size,
        "slide_stride": args.customParameters.stride,
        "comment": "TimeEval execution",
        "seed": args.customParameters.random_state,
        "out_layer_num": args.customParameters.n_out_layers,
        "out_layer_inter_dim": args.customParameters.out_layer_dimensionality,
        "decay": args.customParameters.learning_rate_decay,
        "val_ratio": args.customParameters.split,
        "topk": 20,
    }

    # load data
    env_config = {
        "dataset": ts,
        "dataOutput": args.dataOutput,
        "save_model_path": args.modelOutput,
        "device": args.customParameters.device
    }

    GDNtrain(train_config, env_config)

    # TODO remove
    raise NotImplementedError("GDN is not implemented yet!")


def execute(args: AlgorithmArgs):
    ts = args.ts

    env_config = {
        "dataset": ts,
        "dataOutput": args.dataOutput,
        "load_model_path": args.modelInput,
        "device": args.customParameters.device
    }

    GDNtest(env_config)

    # TODO remove
    raise NotImplementedError("GDN is not implemented yet!")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)
    print(f"AlgorithmArgs: {args}")

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(f"Unknown execution type '{args.executionType}'; expected either 'train' or 'execute'!")
