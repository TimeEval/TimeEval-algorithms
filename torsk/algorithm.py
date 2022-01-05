#!/usr/bin/env python3
import argparse
import json
import logging
import sys
import tarfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any

import numpy as np

import torsk
from torsk.anomaly import sliding_score
from torsk.data.numpy_dataset import NumpyImageDataset as ImageDataset
from torsk.models.numpy_esn import NumpyESN as ESN

np.random.seed(43)
MODEL_PATH = Path("model")


@dataclass
class CustomParameters:
    input_map_size: int = 100
    input_map_scale: float = 0.125
    context_window_size: int = 10  # this is a tumbling window creating the slices
    # -----
    # These create the subsequences (sliding window of train_window_size + prediction_window_size + 1 slices of shape (context_window_size, dim))
    train_window_size: int = 50
    prediction_window_size: int = 20
    transient_window_size: int = 10
    # -----
    spectral_radius: float = 2.0
    density: float = 0.01
    reservoir_representation: str = "sparse"  # sparse is significantly faster
    imed_loss: bool = False  # both work
    train_method: str = "pinv_svd"  # options: "pinv_lstsq", "pinv_svd", "tikhonov"
    tikhonov_beta: Optional[float] = None  # float; only used when train_method="tikhonov"
    verbose: int = 2
    scoring_small_window_size: int = 10
    scoring_large_window_size: int = 100
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
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    return np.genfromtxt(config.dataInput, delimiter=",", skip_header=True)


def configure_logging(config: AlgorithmArgs) -> logging.Logger:
    verbosity = config.customParameters.verbose
    level = "ERROR"
    if verbosity == 1:
        level = "WARNING"
    elif verbosity == 2:
        level = "INFO"
    elif verbosity > 2:
        level = "DEBUG"
    logging.basicConfig(level=level)
    return logging.getLogger(__file__)


def create_torsk_params(config: AlgorithmArgs, shape: np.ndarray) -> torsk.Params:
    # check additional invariants:
    assert config.customParameters.input_map_size >= config.customParameters.context_window_size, \
        "Hidden size must be larger than or equal to input window size!"

    params = torsk.Params()

    # add custom parameters while rewriting some key-names:
    custom_params_dict = asdict(config.customParameters)
    params.train_length = custom_params_dict["train_window_size"]
    params.pred_length = custom_params_dict["prediction_window_size"]
    params.transient_length = custom_params_dict["transient_window_size"]
    del custom_params_dict["train_window_size"]
    del custom_params_dict["prediction_window_size"]
    del custom_params_dict["transient_window_size"]
    del custom_params_dict["random_state"]
    params.update(custom_params_dict)

    # fixed values:
    params.input_map_specs = [{
        "type": "random_weights",
        "size": [config.customParameters.input_map_size],
        "input_scale": config.customParameters.input_map_scale
    }]
    params.input_shape = (config.customParameters.context_window_size, shape[1])
    params.dtype = "float64"  # no need to change
    params.backend = "numpy"  # torch does not work, bh not implemented!
    params.debug = False  # must always be False
    params.timing_depth = config.customParameters.verbose  # verbosity for timing output
    return params


def save_model(model, outfile: Path) -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    torsk.save_model(MODEL_PATH, model)
    # compress folder as archive
    with tarfile.open(outfile, "w:gz") as f:
        f.add(MODEL_PATH)


def load_model(config: AlgorithmArgs) -> Any:
    # decompress archive to folder
    with tarfile.open(config.modelInput, "r:gz") as f:
        f.extractall()

    return torsk.load_model(MODEL_PATH)


def main(config: AlgorithmArgs):
    set_random_state(config)
    logger = configure_logging(config)
    series = load_data(config)[:, 1:-1]

    params = create_torsk_params(config, series.shape)
    logger.info(params)

    logger.info("Using time series dataset ...")
    logger.info(f"{series.shape[0]} {series.shape[1]}-dim. points in series")
    padding_size = 0
    padding_needed = series.shape[0] % params.input_shape[0] != 0
    if padding_needed:
        slices = series.shape[0] // params.input_shape[0]
        padding_size = (slices+1) * params.input_shape[0] - series.shape[0]
        logger.info(f"Series not divisible by context window size, adding {padding_size} padding points")
        series = np.concatenate([series, np.full((padding_size, series.shape[1]), fill_value=0)], axis=0)
    data = series.reshape((series.shape[0] // params.input_shape[0], params.input_shape[0], params.input_shape[1]))
    steps = data.shape[0] - params.train_length - params.pred_length
    dataset = ImageDataset(images=data, scale_images=True, params=params)

    logger.debug(f"Input shape\t: {data.shape}")
    logger.debug(f"Target input shape\t: {params.input_shape}")
    logger.debug(f"Steps\t\t: {steps}")
    logger.debug(f"Dataset length\t: {len(dataset)}")

    logger.info("Building model ...")
    model = ESN(params)

    logger.info("Training + predicting ...")
    predictions, targets = torsk.train_predict_esn(model, dataset, steps=steps, step_length=1, step_start=0)

    try:
        model_output_path = Path(config.modelOutput)
        logger.info(f"Saving model to {model_output_path}")
        save_model(model, model_output_path)
    except AttributeError:
        pass

    logger.info("Calculating anomaly scores ...")
    logger.info(f"Prediction shape={predictions.shape}")
    logger.debug(f"Predictions targets shape={targets.shape}")

    errors = []
    for preds, labels in zip(predictions, targets):
        error = np.abs(labels - preds).mean(axis=-1).mean(axis=0)
        errors.append(error)
    logger.debug(f"{len(predictions)}x error shape: {error.shape}")
    scores, _, _, _ = sliding_score(np.array(errors),
                                    small_window=config.customParameters.scoring_small_window_size,
                                    large_window=config.customParameters.scoring_large_window_size)
    scores = np.concatenate([
        # the first batch of training samples has no predictions --> no scores
        np.full(shape=(params.train_length, config.customParameters.context_window_size), fill_value=np.nan),
        scores
    ], axis=0)
    scores = 1 - scores.ravel()
    if padding_needed:
        # remove padding points
        logger.info("Removing padding from scores ...")
        scores = scores[:-padding_size]
    logger.info(f"Scores (shape={scores.shape}): {scores}")
    np.savetxt(config.dataOutput, scores, delimiter=",")

    # plot(series, scores, config)


def plot(series, scores, config: AlgorithmArgs):
    import matplotlib.pyplot as plt
    from timeeval.utils.window import ReverseWindowing

    def _post_torsk(scores: np.ndarray, pred_size: int = 20, context_window_size: int = 10) -> np.ndarray:
        size = pred_size * context_window_size + 1
        return ReverseWindowing(window_size=size).fit_transform(scores)

    scores_post = _post_torsk(scores, config.customParameters.prediction_window_size, config.customParameters.context_window_size)
    fig, ax = plt.subplots()
    ax.plot(series[:, 0], label="series", color="blue", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax2 = ax.twinx()
    ax2.plot(scores, label="window-score", color="red", alpha=0.25)
    ax2.plot(scores_post, label="point-score", color="green", alpha=0.7)
    ax2.set_ylabel("score")
    plt.legend()
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
        main(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected either 'train' or 'execute'!")
