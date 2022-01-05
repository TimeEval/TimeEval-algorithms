import numpy as np
import pandas as pd
import json
import sys
from dataclasses import dataclass, asdict
import argparse

from ensemble_gi.model import EnsembleGI


@dataclass
class CustomParameters:
    anomaly_window_size: int = 50
    n_estimators: int = 10
    max_paa_transform_size: int = 20
    max_alphabet_size: int = 10
    selectivity: float = 0.8
    random_state: int = 42
    n_jobs: int = 1
    window_method: str = "sliding"  # or "tumbling", "orig"


class AlgorithmArgs(argparse.Namespace):
    @property
    def ts(self) -> np.ndarray:
        return self.df.values[:, 1]

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


def execute(args: AlgorithmArgs):
    data = args.ts
    egi_kwargs = asdict(args.customParameters)
    window_method = egi_kwargs["window_method"]
    del egi_kwargs["window_method"]
    ensemble_gi = EnsembleGI(**egi_kwargs)
    anomaly_scores = ensemble_gi.detect(data, window_method)
    anomaly_scores.tofile(args.dataOutput, sep="\n")


def test(args: AlgorithmArgs):
    import matplotlib.pyplot as plt
    data = args.ts
    data = data[:2000]
    egi_kwargs = asdict(args.customParameters)
    window_method = egi_kwargs["window_method"]
    del egi_kwargs["window_method"]
    egi = EnsembleGI(**egi_kwargs)

    print("\n## original windowing")
    orig = egi.detect(data)
    print("\n## tumbling windowing")
    tumbling = egi.detect(data, window_method="tumbling")
    print("\n## sliding windowing")
    sliding = egi.detect(data, window_method="sliding")

    plt.Figure()
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(data, label="data")
    axs[0].set_title("Timeseries")
    axs[0].legend()

    axs[1].plot(orig, label="orig. windowing")
    axs[1].plot(tumbling, label="tumbling windowing")
    axs[1].plot(sliding, label="sliding windowing")
    axs[1].set_title("Anomal scores")
    axs[1].legend()

    plt.savefig("fig.pdf")
    plt.show()


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()

    if args.executionType == "train":
        print("This algorithm does not need to be trained!")
    elif args.executionType == "execute":
        execute(args)
    elif args.executionType == "test":
        test(args)
    else:
        raise ValueError(f"No executionType '{args.executionType}' available! Choose either 'train' or 'execute'.")
