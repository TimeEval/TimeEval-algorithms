# -*- coding: utf-8 -*-
from collections import namedtuple
import datetime

from pandas import DataFrame, isna
import numpy as np

from s_h_esd.date_utils import format_timestamp, get_gran, date_format
from s_h_esd.detect_anoms import detect_anoms

Direction = namedtuple('Direction', ['one_tail', 'upper_tail'])

def detect_ts(df, max_anoms=0.10, direction='pos',
              alpha=0.05, longterm=False,
              piecewise_median_period_weeks=2, verbose=False):
    """
    Anomaly Detection Using Seasonal Hybrid ESD Test
    A technique for detecting anomalies in seasonal univariate time series where the input is a
    series of <timestamp, value> pairs.
    Args:
    x: Time series as a two column data frame where the first column consists of the
    timestamps and the second column consists of the observations.
    max_anoms: Maximum number of anomalies that S-H-ESD will detect as a percentage of the
    data.
    direction: Directionality of the anomalies to be detected. Options are: ('pos' | 'neg' | 'both').
    alpha: The level of statistical significance with which to accept or reject anomalies.
    longterm: Increase anom detection efficacy for time series that are greater than a month.
    See Details below.
    piecewise_median_period_weeks: The piecewise median time window as described in Vallis, Hochenbaum, and Kejariwal (2014). Defaults to 2.
    Details
    'longterm' This option should be set when the input time series is longer than a month.
    The option enables the approach described in Vallis, Hochenbaum, and Kejariwal (2014).
    The returned value is a dictionary with the following components:
      anoms: Data frame containing timestamps, values, and optionally expected values.
    """

    if not isinstance(df, DataFrame):
        raise ValueError("data must be a single data frame.")
    else:
        if len(df.columns) != 2 or not df.iloc[:,1].map(np.isreal).all():
            raise ValueError(("data must be a 2 column data.frame, with the"
                              "first column being a set of timestamps, and "
                              "the second coloumn being numeric values."))

        if (not (df.dtypes[0].type is np.datetime64)
            and not (df.dtypes[0].type is np.int64)):
            df = format_timestamp(df)

    if list(df.columns.values) != ["timestamp", "value"]:
        df.columns = ["timestamp", "value"]

    # Sanity check all input parameters
    if max_anoms > 0.49:
        length = len(df.value)
        raise ValueError(
            ("max_anoms must be less than 50% of "
             "the data points (max_anoms =%f data_points =%s).")
                         % (round(max_anoms * length, 0), length))

    if not direction in ['pos', 'neg', 'both']:
        raise ValueError("direction options are: pos | neg | both.")

    if not (0.01 <= alpha or alpha <= 0.1):
        if verbose:
            import warnings
            warnings.warn(("alpha is the statistical signifigance, "
                           "and is usually between 0.01 and 0.1"))

    if not isinstance(longterm, bool):
        raise ValueError("longterm must be a boolean")

    if piecewise_median_period_weeks < 2:
        raise ValueError(
            "piecewise_median_period_weeks must be at greater than 2 weeks")

    gran = get_gran(df)

    if gran == 'sec':
        df.timestamp = date_format(df.timestamp.map(lambda d: ":".join(str(d).split(":")[:-1] + ["00"])), "%Y-%m-%d %H:%M:00")
        df = format_timestamp(df.groupby('timestamp').aggregate(np.sum).reset_index())

    # if the data is daily, then we need to bump
    # the period to weekly to get multiple examples
    gran_period = {
        'min': 1440,
        'hr': 24,
        'day': 7
    }
    period = gran_period.get(gran)
    if not period:
        raise ValueError('%s granularity detected. This is currently not supported.' % gran)
    num_obs = len(df.value)

    clamp = (1 / float(num_obs))
    if max_anoms < clamp:
        max_anoms = clamp

    if longterm:
        if gran == "day":
            num_obs_in_period = period * piecewise_median_period_weeks + 1
            num_days_in_period = 7 * piecewise_median_period_weeks + 1
        else:
            num_obs_in_period = period * 7 * piecewise_median_period_weeks
            num_days_in_period = 7 * piecewise_median_period_weeks

        last_date = df.timestamp.iloc[-1]

        all_data = []

        for j in range(0, len(df.timestamp), num_obs_in_period):
            start_date = df.timestamp.iloc[j]
            end_date = min(start_date
                           + datetime.timedelta(days=num_days_in_period),
                           df.timestamp.iloc[-1])

            # if there is at least 14 days left, subset it,
            # otherwise subset last_date - 14days
            if (end_date - start_date).days == num_days_in_period:
                sub_df = df[(df.timestamp >= start_date)
                            & (df.timestamp < end_date)]
            else:
                sub_df = df[(df.timestamp >
                     (last_date - datetime.timedelta(days=num_days_in_period)))
                    & (df.timestamp <= last_date)]
            all_data.append(sub_df)
    else:
        all_data = [df]

    all_anoms = DataFrame(columns=['timestamp', 'value'])
    seasonal_plus_trend = DataFrame(columns=['timestamp', 'value'])
    timestamp_ordering = {}

    # Detect anomalies on all data (either entire data in one-pass,
    # or in 2 week blocks if longterm=TRUE)
    for i in range(len(all_data)):
        directions = {
            'pos': Direction(True, True),
            'neg': Direction(True, False),
            'both': Direction(False, True)
        }
        anomaly_direction = directions[direction]

        # detect_anoms actually performs the anomaly detection and
        # returns the results in a list containing the anomalies
        # as well as the decomposed components of the time series
        # for further analysis.

        s_h_esd_timestamps = detect_anoms(all_data[i], k=max_anoms, alpha=alpha,
                                          num_obs_per_period=period,
                                          use_decomp=True,
                                          one_tail=anomaly_direction.one_tail,
                                          upper_tail=anomaly_direction.upper_tail,
                                          verbose=verbose)

        # store decomposed components in local variable and overwrite
        # s_h_esd_timestamps to contain only the anom timestamps
        data_decomp = s_h_esd_timestamps['stl']
        s_h_esd_timestamps = s_h_esd_timestamps['anoms']
        # -- Step 3: Use detected anomaly timestamps to extract the actual
        # anomalies (timestamp and value) from the data
        if s_h_esd_timestamps:
            anoms = all_data[i][all_data[i].timestamp.isin(s_h_esd_timestamps)]
            timestamp_ordering.update({t:len(s_h_esd_timestamps) - i for i, t in enumerate(s_h_esd_timestamps)})
        else:
            anoms = DataFrame(columns=['timestamp', 'value'])

        all_anoms = all_anoms.append(anoms)
        seasonal_plus_trend = seasonal_plus_trend.append(data_decomp)

    # Cleanup potential duplicates
    try:
        all_anoms.drop_duplicates(subset=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(subset=['timestamp'], inplace=True)
    except TypeError:
        all_anoms.drop_duplicates(cols=['timestamp'], inplace=True)
        seasonal_plus_trend.drop_duplicates(cols=['timestamp'], inplace=True)

    # Calculate number of anomalies as a percentage
    anom_pct = (len(df.value) / float(num_obs)) * 100

    if anom_pct == 0:
        return None

    # left join the original time series with the anomalies
    anoms = df.merge(all_anoms, how="left", on="timestamp")

    # assign each data point an anomaly score based on the ordering of the anom. timestamps
    get_score = lambda row: 0 if isna(row["value_y"]) else timestamp_ordering[row["timestamp"]]
    anoms["value_y"] = anoms.apply(get_score, axis=1)
    anoms.rename(columns={"value_y": "scores"}, inplace=True)
    return anoms
