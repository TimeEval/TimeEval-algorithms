import numpy as np


def original_discretization(X: np.ndarray, l: int, a: int) -> str:  # l=window_size, a=alphabet_size
    features = np.zeros(len(X) - l + 1)
    for i in range(0, len(X) - l + 1):
        features[i] = extract_features(X[i:i+l])
    bins = np.linspace(np.min(features), np.max(features), a)
    x = np.digitize(features, bins)
    charify = np.vectorize(lambda x: chr(x))
    return ''.join(charify(x.astype(int)))


def extract_features(X: np.ndarray) -> float:
    # slope of best fitting line
    coeffs = np.polyfit(np.arange(len(X)), X, 1)
    return coeffs[0].item()
