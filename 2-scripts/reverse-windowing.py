#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def _create_arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate anomaly point scores from anomaly window scores"
    )
    parser.add_argument("-i", "--in-file", type=Path, required=False,
                        default="results/scores.csv",
                        help="File path to the window scores")
    parser.add_argument("-o", "--out-file", type=Path, required=False, default=None,
                        help="File path where the point scores should be written to")
    parser.add_argument("-w", "--window-size", type=int, required=True,
                        help="Window size")
    return parser.parse_args()


def _reverse_windowing(scores: np.ndarray, window_size: int) -> np.ndarray:
    unwindowed_length = (window_size - 1) + len(scores)
    mapped = np.full(shape=(unwindowed_length, window_size), fill_value=np.nan)
    mapped[:len(scores), 0] = scores

    for w in range(1, window_size):
        mapped[:, w] = np.roll(mapped[:, 0], w)

    return np.nanmean(mapped, axis=1)


def reverse(in_path: Path, out_path: Path, window_size: int) -> np.ndarray:
    print(f"Reversing scores from '{in_path}' with window_size={window_size} and writing to '{out_path}'")
    scores = pd.read_csv(in_path, header=None).values[:, 0]
    print(f"Input shape: {scores.shape}")
    scores = _reverse_windowing(scores, window_size)
    print(f"Output shape: {scores.shape}")
    with out_path.open("w") as fh:
        np.savetxt(fh, scores, delimiter=",", newline="\n")


if __name__ == "__main__":
    args = _create_arg_parser()
    if args.out_file is None:
        args.out_file = args.in_file.parent / f"reverse-{args.in_file.name}"
    reverse(args.in_file, args.out_file, window_size=args.window_size)
