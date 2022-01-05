#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
import pyspot as ps

from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class CustomParameters:
    q: float = 1e-3
    n_init: int = 1000
    level: float = 0.99
    max_excess: int = 200
    up: bool = True
    down: bool = True
    alert: bool = True
    bounded: bool = True
    random_state: int = 42


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def from_sys_args() -> 'AlgorithmArgs':
        args: dict = json.loads(sys.argv[1])
        custom_parameter_keys = dir(CustomParameters())
        filtered_parameters = dict(filter(lambda x: x[0] in custom_parameter_keys, args.get("customParameters", {}).items()))
        args["customParameters"] = CustomParameters(**filtered_parameters)
        return AlgorithmArgs(**args)


def load_data(config: AlgorithmArgs) -> np.ndarray:
    df = pd.read_csv(config.dataInput)
    return df.iloc[:, 1].values


def set_random_state(config: AlgorithmArgs) -> None:
    seed = config.customParameters.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


def main(config: AlgorithmArgs):
    assert config.customParameters.n_init * (1 - config.customParameters.level) > 10, "too few data for calibration; either increase n_init or reduce level"
    set_random_state(config)
    data = load_data(config)

    spot = ps.Spot(
        q=config.customParameters.q,
        n_init=config.customParameters.n_init,
        level=config.customParameters.level,
        max_excess=config.customParameters.max_excess,
        up=config.customParameters.up,
        down=config.customParameters.down,
        alert=config.customParameters.alert,
        bounded=config.customParameters.bounded,
    )

    # result types:
    # ------------------------------------------------
    # result    integer	meaning
    # ------------------------------------------------
    # NORMAL	    0	Normal data
    # ALERT_UP  	1	Abnormal data (too high)
    # ALERT_DOWN	-1	Abnormal data (too low)
    # EXCESS_UP	    2	Excess (update the up model)
    # EXCESS_DOWN	-2	Excess (update the down model)
    # INIT_BATCH	3	Data for initial batch
    # CALIBRATION	4	Calibration step
    # ------------------------------------------------
    events = np.zeros_like(data)
    for i, r in enumerate(data):
        event = spot.step(r)
        if event in [1, -1]:
            events[i] = 1

    np.savetxt(config.dataOutput, events, delimiter=",")
    # plot_results(data, events)


def plot_results(values, scores):
    # Plotting stuff
    print("Plotting results to fig.pdf")
    plt.plot(values, lw=2, color="#1B4B5A", label="Data")
    # f1, = plt.plot(up_threshold, ls='dashed', color="#AE81FF", lw=2)
    anomalies_x = []
    anomalies_y = []
    for i, score in enumerate(scores):
        if score > 0:
            anomalies_x.append(i)
            anomalies_y.append(values[i])
    f2 = plt.scatter(anomalies_x, anomalies_y, color="#F55449", label="Detected anomalies")
    plt.legend()
    plt.xlabel("Index / Time")
    plt.ylabel("Values")
    plt.savefig("fig.pdf")


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
