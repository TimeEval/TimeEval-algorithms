import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DAMPPreprocessor(BaseEstimator, TransformerMixin):
    """
    This preprocessing is not part of the paper, but it is used in the authors' code.
    """

    def __init__(self, m: int, sp_index: int):
        self.m = m
        self.sp_index = sp_index

    def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
        X = self._make_2d(X)
        if self._contains_constant_regions(X):
            X += np.arange(len(X)).reshape(-1, 1) / len(X)

        return X

    def _make_2d(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim < 1 or X.ndim > 2:
            raise ValueError(f"The preprocessor can handle only array with a dimensionality of 1 or 2")
        return X

    def _contains_constant_regions(self, X: np.ndarray) -> bool:
        one_row = np.ones(X.shape[1]).reshape(1, -1)
        constant_bool = np.diff(np.concatenate([one_row, np.diff(X, axis=0)], axis=0), axis=0) != 0
        for i in range(X.shape[1]):
            constant_indices = np.argwhere(constant_bool[:, i])
            constant_length = np.max(np.diff(constant_indices, axis=0))
            if constant_length >= self.m or np.var(X[:, i]) < 0.2:
                return True
        return False


if __name__ == "__main__":
    X = np.array([[1,2,2,2,4,5,6.], [1,2,3,3,3,5,6]]).transpose()
    print(X)
    X = DAMPPreprocessor(2).fit_transform(X)
    print("transform...")
    print(X)
