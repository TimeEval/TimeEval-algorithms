import argparse
import json
import sys
import numpy as np
import pandas as pd
from model import AutoEn
from dataclasses import dataclass, asdict


@dataclass
class CustomParameters:
    latent_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.005
    early_stopping_delta: float = 1e-2
    early_stopping_patience: int = 10
    split: float = 0.8
    random_state: int = 42


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
    import random, tensorflow
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.random.set_seed(seed)
    # tensorflow.set_random_seed(seed)


def load_data(args):
    df = pd.read_csv(args.dataInput)
    data = df.iloc[:, 1:-1].values
    labels = df.iloc[:, -1].values
    return data, labels


def train(args):
    xtr, ytr = load_data(args)
    ii = (ytr == 0)
    not_anomaly_data = xtr[ii]
    params = asdict(args.customParameters)
    del params["random_state"]
    model = AutoEn(**params)
    model.fit(not_anomaly_data, args.modelOutput)
    model.save(args.modelOutput)


def pred(args):
    xte, _ = load_data(args)
    model = AutoEn.load(args.modelInput)
    pred = model.predict(xte)
    pred = np.mean(np.abs(pred - xte), axis=1)
    np.savetxt(args.dataOutput, pred, delimiter= ",")


if __name__=="__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        pred(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
