import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import Tuple, Any
import json
import sys
import tarfile
from pathlib import Path

import tensorflow as tf
from tensorflow import keras as K
from tfsnippet.modules import Sequential
from tfsnippet.utils import get_variables_as_dict, VariableSaver
tf.disable_v2_behavior()

from donut import complete_timestamp, standardize_kpi
from donut import Donut
from donut import DonutTrainer, DonutPredictor


MODEL_PATH = Path("tf-model")
ARCNAME = "Model"


@dataclass
class CustomParameters:
    window_size: int = 120
    latent_size: int = 5
    regularization: float = 0.001
    linear_hidden_size: int = 100
    epochs: int = 256  # max_epochs
    random_state: int = 42
    use_column_index: int = 0


class AlgorithmArgs(argparse.Namespace):
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


def prepare_data(args: AlgorithmArgs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    df = args.df
    column_index = 0
    if args.customParameters.use_column_index is not None:
        column_index = args.customParameters.use_column_index
    max_column_index = df.shape[1] - 3
    if column_index > max_column_index:
        print(f"Selected column index {column_index} is out of bounds (columns = {df.columns.values}; "
              f"max index = {max_column_index} [column '{df.columns[max_column_index + 1]}'])! "
              "Using last channel!", file=sys.stderr)
        column_index = max_column_index
    # jump over index column (timestamp)
    column_index += 1

    timestamp, missing, (values, labels) = complete_timestamp(df.timestamp, (df.iloc[:, column_index], df.is_anomaly))
    kpi, mean, std = standardize_kpi(values)
    return kpi, labels, missing, mean, std


def save_model(model_vs, args: AlgorithmArgs):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    var_dict = get_variables_as_dict(model_vs)
    saver = VariableSaver(var_dict, MODEL_PATH)
    saver.save()

    # write archive with model files
    with tarfile.open(args.modelOutput, "w:gz") as f:
        f.add(MODEL_PATH)


def load_model(model_vs, args: AlgorithmArgs):
    # decompress archive with model files
    with tarfile.open(args.modelInput, "r:gz") as f:
        f.extractall()

    saver = VariableSaver(get_variables_as_dict(model_vs), MODEL_PATH)
    saver.restore()


def build_model(args: AlgorithmArgs) -> Tuple[Donut, Any]:
    with tf.variable_scope('model') as model_vs:
        model = Donut(
            h_for_p_x=Sequential([
                K.layers.Dense(args.customParameters.linear_hidden_size,
                               kernel_regularizer=K.regularizers.l2(args.customParameters.regularization),
                               activation=tf.nn.relu),
                K.layers.Dense(args.customParameters.linear_hidden_size,
                               kernel_regularizer=K.regularizers.l2(args.customParameters.regularization),
                               activation=tf.nn.relu),
            ]),
            h_for_q_z=Sequential([
                K.layers.Dense(args.customParameters.linear_hidden_size,
                               kernel_regularizer=K.regularizers.l2(args.customParameters.regularization),
                               activation=tf.nn.relu),
                K.layers.Dense(args.customParameters.linear_hidden_size,
                               kernel_regularizer=K.regularizers.l2(args.customParameters.regularization),
                               activation=tf.nn.relu),
            ]),
            x_dims=args.customParameters.window_size,
            z_dims=args.customParameters.latent_size,
        )
    return model, model_vs


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def train(args: AlgorithmArgs, kpi: np.ndarray, labels: np.ndarray, missing: np.ndarray, mean: float, std: float):
    with tf.Session().as_default():
        def save_callback():
            save_model(model_vs, args)

        model, model_vs = build_model(args)
        trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=args.customParameters.epochs)
        trainer.fit(kpi, labels, missing, mean, std, save_callback=save_callback)
        save_model(model_vs, args)


def execute(args: AlgorithmArgs, kpi: np.ndarray, _labels: np.ndarray, _missing: np.ndarray, _mean: float, _std: float):
    with tf.Session().as_default():
        model, model_vs = build_model(args)
        # initializes variables so they can be filled when loading the model from file
        DonutTrainer(model=model, model_vs=model_vs)
        load_model(model_vs, args)
        predictor = DonutPredictor(model=model)
        # negate as recommended in predictor.get_score(...)
        scores = predictor.get_score(kpi) * -1
    scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)
    kpi = prepare_data(args)

    if args.executionType == "train":
        train(args, *kpi)
    elif args.executionType == "execute":
        execute(args, *kpi)
    else:
        ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
