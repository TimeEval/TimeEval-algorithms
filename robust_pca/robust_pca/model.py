import numpy as np
from sklearn.decomposition import PCA
from typing import Optional

from .r_pca import R_pca


class AnomalyDetector:
    def __init__(self, max_iter: int = 1000):
        self.pca: Optional[PCA] = None
        self.max_iter = max_iter

    def fit(self, X):
        rpca = R_pca(X)
        L, S = rpca.fit(max_iter=self.max_iter)
        self.pca = PCA(n_components=L.shape[1])
        self.pca.fit(L)

    def detect(self, X) -> np.ndarray:
        assert self.pca, "Please train PCA before running the detection!"

        L = self.pca.transform(X)
        S = np.absolute(X - L)
        return S.sum(axis=1)
