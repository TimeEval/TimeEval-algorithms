#!/usr/bin/env python3
import json
import sys

from pathlib import Path

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise ValueError("You have to specify an algorithm name (directory / docker image name)!")

    algorithm = args[1]
    manifest_path = Path(".") / algorithm / "manifest.json"
    with manifest_path.open("r") as fh:
        manifest = json.load(fh)

    value = manifest["inputDimensionality"]
    if value.lower() == "univariate":
        print("data/dataset.csv")
    elif value.lower() == "multivariate":
        print("data/multi-dataset.csv")
    else:
        raise ValueError(f"Input dimensionality ({value}) of {algorithm}'s manifest is unknown!")
