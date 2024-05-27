#!/usr/bin/env python
import sys, json
import numpy as np
from series2graph import Series2Graph
from pathlib import Path


class Config:
    dataInput: Path
    dataOutput: Path
    executionType: str
    l: int
    ql: int
    latent: int
    rate: int
    random_state: int

    def __init__(self, params):
        self.dataInput = Path(params.get("dataInput", "/data/dataset.csv"))
        self.dataOutput = Path(params.get("dataOutput", "/results/anomaly_window_scores.ts"))
        self.executionType = params.get("executionType", "execute")
        # ignore modelInput and modelOutput, because it is not needed
        try:
            customParameters = params["customParameters"]
        except KeyError:
            customParameters = {}
        self.l = customParameters.get("window_size", 50)
        self.ql = customParameters.get("query_window_size", 75)
        self.latent = self.l // 3
        self.rate = customParameters.get("rate", 30)
        self.random_state = customParameters.get("random_state", 42)
    
    def __str__(self):
        return f"Config("\
            f"dataInput={self.dataInput}, dataOutput={self.dataOutput}, executionType={self.executionType}," \
            f"l={self.l}, ql={self.ql}, latent={self.latent}, rate={self.rate})"


class TS():
    def __init__(self, vs):
        self.values = vs

    def __repr__(self):
        return f"TS({self.values})"


def load_ts(filename):
    values = np.genfromtxt(filename, skip_header=1, delimiter=",", usecols=(1,))
    return [TS(values)]


def main(config):
    ts = load_ts(config.dataInput)
    print(f"Read input time series from {config.dataInput}:", ts)

    s2g = Series2Graph(pattern_length=config.l, latent=config.latent, rate=config.rate)
    s2g.fit(ts)
    s2g.score(query_length = config.ql)

    print("len(ts):", len(ts[0].values))
    print("len(scores):", len(s2g.all_score))
    print(f"Anomaly window scores written to {config.dataOutput}:", s2g.all_score)
    np.savetxt(config.dataOutput, s2g.all_score, delimiter=",")


def parse_args():
    if len(sys.argv) < 2:
        print("No arguments supplied, using default arguments!", file=sys.stderr)
        params = {}
    elif len(sys.argv) > 2:
        print("Wrong number of arguments supplied! Single JSON-String expected!", file=sys.stderr)
        exit(1)
    else:
        params = json.loads(sys.argv[1])
    return Config(params)


def set_random_state(config) -> None:
    seed = config.random_state
    import random
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    config = parse_args()
    print(config)
    set_random_state(config)
    if config.executionType == "train":
        print("No training required!")
        exit(0)
    else:
        main(config)
