import argparse
from dataclasses import dataclass
import sys
import json

import numpy as np
import pandas as pd

from s_h_esd.detect_ts import detect_ts

@dataclass
class CustomParameters:
    max_anomalies: float = 0.05
    timestamp_unit: str = "m"  # or "h", "d"
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


def load_data(path: str) -> np.ndarray:
    return pd.read_csv(path, parse_dates=["timestamp"], infer_datetime_format=True).iloc[:, :2]


def execute(config):
    ts: pd.DataFrame = load_data(config.dataInput)
    if ts.dtypes["timestamp"] != np.datetime64:
        print(f"Converting index column 'timestamp' to np.datetime64 assuming unit '{config.customParameters.timestamp_unit}'")
        ts["timestamp"] = pd.to_datetime(ts["timestamp"].astype(int),
                                         unit=config.customParameters.timestamp_unit,
                                         origin=pd.Timestamp("1970-01-01 00:00"))

    # clip max_anomalies to prevent error:
    # With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have 10028 observations in a period, which is too few. Set a higher piecewise_median_period_weeks.
    max_anomalies = config.customParameters.max_anomalies
    if max_anomalies < 0.001:
        print(f"WARN: max_anomalies (={max_anomalies}) must be at least 0.001! Dynamically fixing this by setting max_anomalies to 0.001")
        max_anomalies = 0.001

    # we convert the integer time index to timestamps and according to the documentation the
    # longterm option should be set, when > 1 month
    duration = ts["timestamp"].max() - ts["timestamp"].min()
    longterm = False
    # 2628000 seconds in one month
    if duration.total_seconds() > 2628000:
        longterm = True
    print(f"Duration of TS={duration} > 1 month?\n\t--> longterm-support {'enabled' if longterm else 'disabled'}")

    print("Detecting anomalies...")
    anomalies = detect_ts(ts,
        max_anoms=max_anomalies,
        longterm=longterm,
        direction="both",
    )
    np.savetxt(config.dataOutput, anomalies["scores"].values)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong number of arguments specified; expected a single json-string!")
        exit(1)

    config = AlgorithmArgs.from_sys_args()
    print(config)
    set_random_state(config)

    if config.executionType == "train":
        print("Nothing to train, finished!")
        exit(0)
    elif config.executionType == "execute":
        execute(config)
    else:
        raise ValueError(f"Unknown execution type '{config.executionType}'; expected 'execute'!")
