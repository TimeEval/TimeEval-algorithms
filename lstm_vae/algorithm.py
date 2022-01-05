import sys
import argparse
import json
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataclasses import dataclass
from joblib import dump, load
import pandas as pd
from src.model import LSTM_VAE


@dataclass
class CustomParameters:
    learning_rate: int = 0.001
    epochs: int = 10
    batch_size: int = 32
    window_size: int = 10
    latent_size: int = 5
    lstm_layers: int = 10
    rnn_hidden_size: int = 5
    early_stopping_delta: float = 0.05
    early_stopping_patience: int = 10



class AlgorithmArgs(argparse.Namespace):
    @property
    def df(self) -> pd.DataFrame:
        df = pd.read_csv(self.dataInput).drop(["is_anomaly", "timestamp"], axis=1, errors="ignore")
        if df.shape[1] == 1:
            return df
        else:
            return df.iloc[:, 1]


    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        if len(sys.argv) != 2:
            raise ValueError("Wrong number of arguments specified! Single JSON-string pos. argument expected.")
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def save_model(model: LSTM_VAE, args: AlgorithmArgs):
    dump(model, args.modelOutput)


def load_model(args: AlgorithmArgs):
    model = load(args.modelInput)
    return model


def train(args: AlgorithmArgs):
    df = args.df

    print("Executing LSTM-VAE...")
    model = LSTM_VAE(window_size=args.customParameters.window_size, 
                    lstm_layers=args.customParameters.lstm_layers,
                    rnn_hidden_size=args.customParameters.rnn_hidden_size, 
                    latent_size=args.customParameters.latent_size,
                    early_stopping_patience=CustomParameters.early_stopping_patience,
                    early_stopping_delta=CustomParameters.early_stopping_delta,
                    input_size=df.shape[1])

    optimizer = Adam(model.parameters(), lr=args.customParameters.learning_rate)

    df = model.prepare_data_train(df=df)
    train_loader = DataLoader(dataset=df, 
                            batch_size=args.customParameters.batch_size, 
                            shuffle=True)
    model.fit(optimizer, args.customParameters.epochs, train_loader, args.modelOutput)
    save_model(model, args)


def execute(args: AlgorithmArgs):
    df = args.df

    model = load_model(args)
    df = model.prepare_data_execute(df=df)

    exec_loader = DataLoader(dataset=df, 
                            batch_size=CustomParameters.batch_size, 
                            shuffle=False)
    print("Predicting...")
    scores = model.detect(exec_loader)
    scores.tofile(args.dataOutput, sep="\n")


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        train(args)
    elif args.executionType == "execute":
        execute(args)
    else:
        ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
