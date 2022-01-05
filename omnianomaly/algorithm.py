import pandas as pd
import numpy as np
import argparse
import json
import sys
from pathlib import Path
import tarfile

import tensorflow as tf

from omni_anomaly.prediction import Predictor
from omni_anomaly.model import OmniAnomaly
from omni_anomaly.training import Trainer
from tfsnippet.scaffold import VariableSaver
from tfsnippet.utils import get_variables_as_dict

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIR = Path("./tf-model")


class CustomParameters:
    def __init__(self, **params):
        self.dictionary = {
            "use_connected_z_q": True,
            "use_connected_z_p": True,
            # model parameters
            "x_dim": 1,
            "z_dim": 3,
            "rnn_cell": 'GRU',  # 'GRU', 'LSTM' or 'Basic'
            "rnn_num_hidden": 500,
            "window_size": 100,
            "dense_dim": 500,
            "posterior_flow_type": 'nf',  # 'nf' or None
            "nf_layers": 20,  # for nf
            "epochs": 10,  # max_epoch
            "train_start": 0,  # not exposed, always from beginning!
            "test_start": 0,  # not exposed, always from beginning!
            "batch_size": 50,
            "l2_reg": 0.0001,
            "learning_rate": 0.001,
            "lr_anneal_factor": 0.5,
            "lr_anneal_epoch_freq": 40,
            "lr_anneal_step_freq": None,
            "std_epsilon": 1e-4,

            "valid_step_freq": 100,
            "gradient_clip_norm": 10.,

            "early_stop": True,  # whether to apply early stop method
            "level": 0.01,

            "test_n_z": 1,

            "save_z": False,  # whether to save sampled z in hidden space
            "get_score_on_dim": False,  # whether to get score on dim. If `True`, the score will be a 2-dim ndarray
            "save_dir": 'model',
            "restore_dir": None,  # If not None, restore variables from this dir
            "result_dir": '.',  # Where to save the result file
            "train_score_filename": 'train_score.pkl',
            "test_score_filename": 'test_score.pkl',
            "random_state": 42,
            "split": 0.8
        }

        def set_renamed_param(params, k_orig, k_new):
            if k_new in params:
                self.dictionary[k_orig] = params[k_new]

        self.dictionary = dict((k, params.get(k, v)) for k, v in self.dictionary.items())
        set_renamed_param(params, "z_dim", "latent_size")
        set_renamed_param(params, "rnn_num_hidden", "rnn_hidden_size")
        set_renamed_param(params, "dense_dim", "linear_hidden_size")

        for k, v in self.dictionary.items():
            setattr(self, k, v)


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        dataset = pd.read_csv(self.dataInput)
        return dataset.values[:, 1:-1]

    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        args["customParameters"] = CustomParameters(**args.get("customParameters", {}))
        return AlgorithmArgs(**args)


class OldTFModelSaver:
    def __init__(self, model_vs, args: AlgorithmArgs, model_dir: Path = MODEL_DIR):
        self.model_vs = model_vs
        self.args = args
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save(self):
        var_dict = get_variables_as_dict(self.model_vs)
        saver = VariableSaver(var_dict, self.model_dir)
        saver.save()

        # write archive with model files
        with tarfile.open(self.args.modelOutput, "w:gz") as f:
            f.add(self.model_dir)

    def load(self):
        # decompress archive with model files
        with tarfile.open(self.args.modelInput, "r:gz") as f:
            f.extractall()

        saver = VariableSaver(get_variables_as_dict(self.model_vs), self.model_dir)
        saver.restore()


def prepare_data(args: AlgorithmArgs) -> np.ndarray:
    ts = args.ts
    args.customParameters.x_dim = ts.shape[1]
    return ts


def train(args: AlgorithmArgs):
    ts = prepare_data(args)
    config = args.customParameters

    with tf.Session().as_default():
        with tf.variable_scope('model') as model_vs:
            def save():
                return OldTFModelSaver(model_vs, args).save()

            model = OmniAnomaly(config=config, name="model")
            trainer = Trainer(model=model,
                              model_vs=model_vs,
                              max_epoch=config.epochs,
                              batch_size=config.batch_size,
                              valid_batch_size=config.batch_size,
                              initial_lr=config.learning_rate,
                              lr_anneal_epochs=config.lr_anneal_epoch_freq,
                              lr_anneal_factor=config.lr_anneal_factor,
                              grad_clip_norm=config.gradient_clip_norm,
                              valid_step_freq=config.valid_step_freq)
            trainer.fit(ts, valid_portion=1-config.split, with_stats=False, save_model_fn=save)

            save()


def execute(args: AlgorithmArgs):
    ts = prepare_data(args)
    config = args.customParameters

    with tf.Session().as_default():
        with tf.variable_scope('model') as model_vs:
            model = OmniAnomaly(config=config, name="model")
            # initializes variables so they can be filled when loading the model from file
            Trainer(model=model,
                    model_vs=model_vs,
                    max_epoch=config.epochs,
                    batch_size=config.batch_size,
                    valid_batch_size=config.batch_size,
                    initial_lr=config.learning_rate,
                    lr_anneal_epochs=config.lr_anneal_epoch_freq,
                    lr_anneal_factor=config.lr_anneal_factor,
                    grad_clip_norm=config.gradient_clip_norm,
                    valid_step_freq=config.valid_step_freq)

            OldTFModelSaver(model_vs, args).load()

            predictor = Predictor(model, batch_size=config.batch_size, n_z=config.test_n_z, last_point_only=True)
            # negate as recommended in predictor.get_score(...)
            scores = predictor.get_score(ts, with_stats=False) * -1
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
