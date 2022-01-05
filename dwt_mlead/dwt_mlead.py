import warnings
from typing import Callable, List, NamedTuple, Optional, Any

import numpy as np
import pywt as wt
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import DBSCAN
from sklearn.covariance import EmpiricalCovariance

warnings.filterwarnings(action='ignore', category=UserWarning)


class AnomalyCluster(NamedTuple):
    center: float
    score: float
    points: np.ndarray


def pad_series(data: np.ndarray) -> np.ndarray:
    n = len(data)
    exp = np.ceil(np.log2(n))
    m = int(np.power(2, exp))
    return wt.pad(data, (0, m - n), "periodic")


def multilevel_dwt(data: np.ndarray, wavelet: str = "haar", mode: str = "periodic", level_from: int = 0, level_to=None):
    if level_to is None:
        level_to = int(np.log2(len(data)))
    ls_ = []
    as_ = []
    ds_ = []
    a = data
    for i in range(level_to):
        a, d = wt.dwt(a, wavelet, mode)
        if i + 1 >= level_from:
            ls_.append(i + 1)
            as_.append(a)
            ds_.append(d)
    return np.array(ls_), as_, ds_


def reverse_windowing(data: np.ndarray, window_length: int, full_length: int,
                      reduction: Callable = np.mean,
                      fill_value: float = np.nan) -> np.ndarray:
    mapped = np.full(shape=(full_length, window_length), fill_value=fill_value)
    mapped[:len(data), 0] = data

    for w in range(1, window_length):
        mapped[:, w] = np.roll(mapped[:, 0], w)

    return reduction(mapped, axis=1)


def combine_alternating(xs, ys):
    for x, y in zip(xs, ys):
        yield x
        yield y


class DWT_MLEAD():
    def __init__(self, data: np.ndarray, start_level: int, quantile_boundary_type: str, quantile_epsilon: float,
                 track_coefs: bool = False):
        self.n = len(data)
        self.data = pad_series(data)
        self.m = len(self.data)
        self.start_level = start_level
        self.max_level = int(np.log2(self.m))
        self.quantile_boundary_type = quantile_boundary_type
        self.quantile_epsilon = quantile_epsilon

        self.window_sizes = np.array(
            [max(2, self.max_level - l - self.start_level + 1) for l in range(self.max_level)])

        self.track_coefs = track_coefs
        self.coefs_levels_: Optional[np.ndarray] = None
        self.coefs_approx_: Optional[List[np.ndarray]] = None
        self.coefs_detail_: Optional[List[np.ndarray]] = None
        self.coefs_scores_: Optional[List[np.ndarray]] = None

    def detect(self) -> np.ndarray:
        levels, approx_coefs, detail_coefs = multilevel_dwt(self.data,
            wavelet="haar",
            mode="periodic",
            level_from=self.start_level,
            # skip last level, because we cannot slide a window of size 2 over it (too small)
            level_to=self.max_level - 1,)

        coef_anomaly_counts = []
        for x, level in zip(combine_alternating(detail_coefs, approx_coefs), levels.repeat(2, axis=0)):
            window_size = self.window_sizes[level]
            x_view = sliding_window_view(x, window_size)

            p = self._estimate_gaussian_likelihoods(level, x_view)
            a = self._mark_anomalous_windows(p)
            xa = reverse_windowing(a, window_length=window_size, full_length=len(x), reduction=np.sum, fill_value=0)
            coef_anomaly_counts.append(xa)
        if self.track_coefs:
            self.coefs_levels_ = levels
            self.coefs_approx_ = approx_coefs
            self.coefs_detail_ = detail_coefs
            self.coefs_scores_ = coef_anomaly_counts

        point_anomaly_scores = self._push_anomaly_counts_down_to_points(coef_anomaly_counts)
        # anomaly_clusters = self.find_cluster_anomalies(point_anomaly_scores)
        return point_anomaly_scores

    def _estimate_gaussian_likelihoods(self, level: float, x_view: np.ndarray) -> np.ndarray:
        # fit gaussion distribution with mean and covariance
        e_cov_est = EmpiricalCovariance(assume_centered=False)
        e_cov_est.fit(x_view)

        # compute log likelihood for each window x in x_view
        p = np.empty(shape=len(x_view))
        for i, window in enumerate(x_view):
            p[i] = e_cov_est.score(window)

        print(f"Gaussion Distribution for level {level}:")
        print(f"Shapes: mean={e_cov_est.location_.shape}, covariance={e_cov_est.covariance_.shape}, p={p.shape}")
        return p

    def _mark_anomalous_windows(self, p: np.ndarray):
        if self.quantile_boundary_type == "percentile":
            z_eps = np.percentile(p, self.quantile_epsilon * 100)
        else:  # self.quantile_boundary_type == "monte-carlo"
            raise ValueError(f"The quantile boundary type '{self.quantile_boundary_type}' is not implemented yet!")

        return p < z_eps

    def _push_anomaly_counts_down_to_points(self, coef_anomaly_counts: List[np.ndarray]) -> np.ndarray:
        # sum up anomaly counters of detail coefs (orig. D^l) and approx coefs (orig. C^l)
        anomaly_counts = coef_anomaly_counts[0::2]
        anomaly_counts += coef_anomaly_counts[1::2]

        # extrapolate anomaly counts to the original series' points
        counter = np.zeros(self.m)
        for ac in anomaly_counts:
            counter += ac.repeat(self.m // len(ac), axis=0)
        # delete event counters with count < 2
        counter[counter < 2] = 0
        return counter[:self.n]

    def find_cluster_anomalies(self, point_anomaly_scores: np.ndarray,
                               d_max: float,
                               anomaly_counter_threshold: float) -> List[AnomalyCluster]:
        indices = np.arange(len(point_anomaly_scores))
        anomalous_point_ids = indices[point_anomaly_scores != 0]

        # clustering
        dbs = DBSCAN(eps=d_max, min_samples=5).fit(anomalous_point_ids.reshape(-1, 1))

        # collecting cluster anomalies
        anomaly_clusters: List[AnomalyCluster] = []
        classes = np.unique(dbs.labels_)
        for i in classes:
            if i != -1:
                cluster_points = indices[anomalous_point_ids[dbs.labels_ == i]]
                cluster_center = int(np.average(cluster_points, weights=point_anomaly_scores[cluster_points]))
                cluster_score = point_anomaly_scores[cluster_points].sum()
                if cluster_score > anomaly_counter_threshold:
                    anomaly_clusters.append(AnomalyCluster(cluster_center, cluster_score, cluster_points))
                else:
                    print(f"Cluster {i} with center {cluster_center} is not anomalous.")
        return anomaly_clusters

    def plot(self, point_anomaly_scores: Optional[np.ndarray] = None,
             anomaly_clusters: Optional[List[AnomalyCluster]] = None,
             coefs: bool = False) -> Any:
        import pandas as pd
        import matplotlib.pyplot as plt

        print("\n=== Plotting results ===")
        print("Collecting data")
        df = pd.DataFrame(self.data[:self.n], columns=["data"])
        if coefs:
            if (self.coefs_levels_ is None or self.coefs_approx_ is None
                    or self.coefs_detail_ is None or self.coefs_scores_ is None):
                import sys
                print(f"Cannot plot coefs, because they were not tracked! "
                      f"Use `track_coefs=True` to enable plotting coefs.", file=sys.stderr)
            else:
                for i, d, a, s in zip(self.coefs_levels_, self.coefs_detail_, self.coefs_approx_, self.coefs_scores_):
                    df[f"DetailCoef Level {i}"] = d.repeat(self.m // len(d), axis=0)[:self.n]
                    df[f"ApproxCoef Level {i}"] = a.repeat(self.m // len(a), axis=0)[:self.n]
                    df[f"Coef Score Level {i}"] = s.repeat(self.m // len(s), axis=0)[:self.n]

        if point_anomaly_scores is not None:
            df["Point Anomaly Score"] = point_anomaly_scores

        if anomaly_clusters is not None:
            df["Cluster"] = -1
            df["Cluster Anomaly Score"] = np.nan
            for i, cluster in enumerate(anomaly_clusters):
                df.loc[cluster.points, "Cluster"] = i
                df.loc[cluster.points, "Cluster Anomaly Score"] = cluster.score

        n_plots = 1 + \
                  (len(self.coefs_levels_) if coefs and self.coefs_levels_ is not None else 0) + \
                  (1 if point_anomaly_scores is not None or anomaly_clusters is not None else 0)
        print(f"Creating {n_plots} subplots")
        fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex=True)

        # plot original data
        print("Plotting original data")
        axs[0].plot(df["data"], label="Original series")
        axs[0].legend()

        # plot coefficients
        if coefs and self.coefs_levels_ is not None:
            print("Plotting DWT coefficients and their anomaly scores")
            for i in self.coefs_levels_:
                for column in [f"DetailCoef Level {i}", f"ApproxCoef Level {i}"]:
                    axs[i - self.start_level + 1].plot(df[column], label=column)
                scale = np.max([df[f"DetailCoef Level {i}"], df[f"ApproxCoef Level {i}"]]) / np.max(
                    df[f"Coef Score Level {i}"])
                axs[i - self.start_level + 1].plot(df[f"Coef Score Level {i}"] * scale, label=f"Coef Score Level {i}")
                axs[i - self.start_level + 1].legend()

        # plot point scores
        if point_anomaly_scores is not None:
            print("Plotting point anomaly scores")
            axs[-1].plot(df["Point Anomaly Score"], label="Point Anomaly Score")

        # plot cluster scores
        if anomaly_clusters is not None:
            print("Plotting anomaly clusters")
            classes = df["Cluster"].unique()
            for i in classes:
                if i != -1:
                    index = df[df["Cluster"] == i].index.values
                    axs[-1].scatter(x=index, y=[1 for x in index], label=f"cluster {i}")

        if point_anomaly_scores is not None or anomaly_clusters is not None:
            axs[-1].legend()

        plt.show()
        return df
