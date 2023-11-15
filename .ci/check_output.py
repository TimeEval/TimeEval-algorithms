#!/usr/bin/env python3
import json
import sys

from pathlib import Path

MODEL_FILEPATH = Path("./results/model.pkl")
SCORES_FILEPATH = Path("./results/scores.csv")


def parse_manifest(algorithm: str) -> dict:
    manifest_path = Path(".") / algorithm / "manifest.json"
    with manifest_path.open("r") as fh:
        manifest = json.load(fh)
    return manifest


def is_readable(filename: Path) -> bool:
    stat = filename.stat()
    return stat.st_uid == 1000 and stat.st_gid == 1000


def has_postprocessing(algorithm: str) -> bool:
    readme_path = Path(".") / algorithm / "README.md"
    if not readme_path.exists():
        return False

    with readme_path.open("r") as fh:
        readme = fh.readlines()

    marker = ["<!--BEGIN:timeeval-post-->", "<!--END:timeeval-post-->"]
    return any([m in l for m in marker for l in readme])


def main(algorithm):
    manifest = parse_manifest(algorithm)
    errors = []

    if manifest["learningType"].lower() in ["supervised", "semi-supervised"]:
        # check model.pkl
        if not is_readable(MODEL_FILEPATH):
            errors.append("Model file was written with the wrong user and/or group. Do you use a TimeEval base image?")

    # check scores.csv
    if not is_readable(SCORES_FILEPATH):
        errors.append("Scoring was written with the wrong user and/or group. Do you use a TimeEval base image?")

    with SCORES_FILEPATH.open("r") as fh:
        lines = fh.readlines()


    # if not post-processing, check length
    if has_postprocessing(algorithm):
        print("Skipping scoring (scores.csv) check, because algorithm uses post-processing!")
    else:
        # only a single column/dimension:
        if any(["," in l for l in lines]):
            errors.append("Scoring contains multiple dimensions (found a ',' in the file). "
                        "Only a single anomaly score is allowed per time step!")

        # there should be no header
        try:
            float(lines[0])
        except ValueError as e:
            errors.append(f"No header allowed for the scoring file! First value is not a number! {e}")

        # same length as dataset
        if manifest["inputDimensionality"].lower() == "univariate":
            data_path = Path("./data/dataset.csv")
        else:
            data_path = Path("./data/multi-dataset.csv")

        n_data = 0
        with data_path.open("r") as fh:
            for _ in fh:
                n_data += 1
        # substract header
        n_data -= 1

        if len(lines) != n_data:
            errors.append("Scoring has wrong length; each input time step needs an anomaly score "
                          f"(expected={n_data}, found={len(lines)})!")

    for error in errors:
        print(error, file=sys.stderr)

    if len(errors) > 0:
        exit(1)


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        raise ValueError("You have to spacify an algorithm name (directory / docker image name)!")

    main(args[1])
