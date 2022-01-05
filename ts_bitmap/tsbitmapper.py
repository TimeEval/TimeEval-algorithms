from math import ceil
import numpy as np
from collections import defaultdict
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa


class TSBitMapper:
    """

    Implements Time-series Bitmap model for anomaly detection

    Based on the papers "Time-series Bitmaps: A Practical Visualization Tool for working with Large Time Series Databases"
    and "Assumption-Free Anomaly Detection in Time Series"

    Test data and parameter settings taken from http://alumni.cs.ucr.edu/~wli/SSDBM05/
    """

    def __init__(self, feature_window_size=None, bins=5, level_size=3, lag_window_size=None, lead_window_size=None,
                 compression_ratio=1):

        """

        :param int feature_window_size: should be about the size at which events happen
        :param int or array-like bins: a scalar number of equal-width bins or a 1-D and monotonic array of bins.
        :param int level_size: desired level of recursion of the bitmap
        :param int lag_window_size: how far to look back, None for supervised learning
        :param int lead_window_size: how far to look ahead
        :param int compression_ratio: how much to compress the timeseries in the paa step
        """

        assert feature_window_size > 0, 'feature_window_size must be a positive integer'
        assert lead_window_size > 0, 'lead_window_size must be a positive integer'
        assert lag_window_size > 0, 'lag_window_size must be a positve integer'
        assert compression_ratio >= 1, 'PAA needs a compression ratio >= 1'

        # bitmap parameters
        self._feature_window_size = feature_window_size
        self._lag_window_size = lag_window_size
        self._lead_window_size = lead_window_size
        self._num_bins = bins
        self._level_size = level_size
        self._compression_ratio = compression_ratio


    def fit_predict(self, ts):
        """
        Unsupervised training of TSBitMaps.

        :param ts: 1-D numpy array or pandas.Series
        :return scores: anomaly score for each sax symbol
        """
        self._ref_ts = ts
        self._ts_length = len(ts)
        scores = self._slide_chunks(ts)
        return scores


    def _slide_chunks(self, ts):
        lag_bitmap = {}
        lead_bitmap = {}

        egress_lag_feat = ()
        egress_lead_feat = ()

        binned_ts = self.discretize_by_sax_window(ts)
        scores = np.zeros(len(binned_ts))
        ts_len = len(binned_ts)

        lagws = self._lag_window_size
        leadws = self._lead_window_size
        featws = self._level_size
        print(f"length after discretization: {ts_len}")
        print(f"sliding lag window (size={lagws}) and lead window (size={leadws}) with depth {featws}")
        for i in range(lagws, ts_len - leadws + 1):
            lag_chunk = binned_ts[i - lagws: i]
            lead_chunk = binned_ts[i: i + leadws]

            if i == lagws:
                lag_bitmap = self.get_bitmap_with_feat_window(lag_chunk)
                lead_bitmap = self.get_bitmap_with_feat_window(lead_chunk)
                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)

            else:
                ingress_lag_feat = lag_chunk[-featws:]
                ingress_lead_feat = lead_chunk[-featws:]

                lag_bitmap[ingress_lag_feat] += 1
                lag_bitmap[egress_lag_feat] -= 1

                lead_bitmap[ingress_lead_feat] += 1
                lead_bitmap[egress_lead_feat] -= 1


                scores[i] = self.bitmap_distance(lag_bitmap, lead_bitmap)
                # print(f"i={i}: ingress(lag={ingress_lag_feat}, lead={ingress_lead_feat}), egress(lag={egress_lag_feat}, lead={egress_lead_feat})")

            egress_lag_feat = lag_chunk[0: featws]
            egress_lead_feat = lead_chunk[0: featws]

            # print(f"i={i}: lag={i - lagws}:{i},  lead={i}:{i + leadws}")

        return scores


    def discretize_by_sax_window(self, ts, feature_window_size=None):
        if feature_window_size is None:
            feature_window_size = self._feature_window_size
        n = len(ts)
        windows = ()
        for i in range(0, n - n % feature_window_size, feature_window_size):
            binned_fw = self.discretize_sax(ts[i: i + feature_window_size])
            windows += binned_fw
        if n % feature_window_size > 0:
            last_binned_fw = self.discretize_sax(ts[- (n % feature_window_size):])
            windows += last_binned_fw
        return windows


    def discretize_sax(self, ts):
        znorm_ts = znorm(ts)
        if self._compression_ratio == 1:
            ts_string = ts_to_string(znorm_ts, cuts=cuts_for_asize(self._num_bins))
        else:
            ts_string = self.apply_paa(znorm_ts)
        return tuple(ts_string)


    def apply_paa(self, ts):
        num_paa_segments = ceil(ts.size / self._compression_ratio)
        paa_ts = paa(ts, paa_segments=num_paa_segments)
        return ts_to_string(paa_ts, cuts=cuts_for_asize(self._num_bins))


    def get_bitmap_with_feat_window(self, chunk, level_size=None, step=None):
        """

        :param str chunk: symbol sequence representation of a univariate time series
        :param int level_size: desired level of recursion of the bitmap
        :param int step: length of the feature window
        :return : bitmap representation of `chunk`
        """
        if step is None:
            step = self._feature_window_size
        if level_size is None:
            level_size = self._level_size
        bitmap = defaultdict(int)
        n = len(chunk)

        for i in range(0, n - n % step, step):

            for j in range(step - level_size + 1):
                feat = chunk[i + j: i + j + level_size]
                bitmap[feat] += 1  # frequency count

        if n % step > 0:
            for i in range(n - n % step, n - level_size + 1):
                feat = chunk[i: i + level_size]
                bitmap[feat] += 1

        max_freq = max(bitmap.values())

        for feat in bitmap.keys():
            bitmap[feat] = bitmap[feat] / max_freq
        return bitmap


    def bitmap_distance(self, lag_bitmap, lead_bitmap):
        """
        Computes the dissimilarity of two bitmaps.
        """
        dist = 0
        lag_feats = set(lag_bitmap.keys())
        lead_feats = set(lead_bitmap.keys())
        shared_feats = lag_feats & lead_feats

        for feat in shared_feats:
            dist += (lead_bitmap[feat] - lag_bitmap[feat]) ** 2

        for feat in lag_feats - shared_feats:
            dist += lag_bitmap[feat] ** 2

        for feat in lead_feats - shared_feats:
            dist += lead_bitmap[feat] ** 2

        return dist


    def get_window_sizes(self, start, stop, step):
        return np.expand_dims(np.diff(np.append(np.arange(start, stop, step), stop)), axis=1)

    """
    example:
    - ts_length: 43
    - window_size: 20
    - compr_ratio: 3
    """
    def post_ts_bitmap(self, scores: np.ndarray):
        if self._compression_ratio == 1:
            return scores
        window_sizes = np.diff(np.arange(0, self._ts_length, self._feature_window_size))
        # window_sizes: [20  20  3]
        mini_window_sizes = self.get_window_sizes(0, self._feature_window_size, self._compression_ratio)
        # mini_window_sizes: [6  6  6  2]
        length_encoding = np.tile(mini_window_sizes, len(window_sizes))
        length_encoding_t = np.transpose(length_encoding)
        # length_encoding_t:
        # [[6  6  6  2],
        #  [6  6  6  2]]
        # last window is still missing (index 40-42)

        last_window_start = np.arange(0, self._ts_length, self._feature_window_size)[-1]
        length_encoding_full = np.append(length_encoding_t,
                                        self.get_window_sizes(last_window_start,
                                                              self._ts_length,
                                                              self._compression_ratio))
        # length_encoding_full:
        # [6  6  6  2  6  6  6  2  3]

        # each score now appears at the same index as its corresponding length in the length encoding
        decompressed_scores = np.zeros(self._ts_length)
        index_decompressed = 0
        for i in range(len(scores)):
            tmp = np.repeat(scores[i], length_encoding_full[i])
            decompressed_scores[index_decompressed: index_decompressed + length_encoding_full[i]] = tmp
            index_decompressed += length_encoding_full[i]

        return decompressed_scores
