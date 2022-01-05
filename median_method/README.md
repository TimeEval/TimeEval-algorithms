# MedianMethod

|||
| :--- | :--- |
| Citekey | BasuMeckesheimer2007Automatic |
| Source Code | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |

## General Notes

This simple method detects outliers in linear time by looking at the neighbourhood of each data point and creating windows of that neighbourhood.
The median of each window is calculated and compared with the current data point.
The corresponding anomaly score results from calculating how many standard deviations the current data point differs from the median of its neighbourhood.

## Custom Parameters

- _neighbourhood_size_: specifies the number of time steps to look forward and backward for each data point

## Citation format (for source code)

> Basu, S., & Meckesheimer, M. (2007). Automatic outlier detection for time series: an application to sensor data. Knowledge and Information Systems, 11(2), 137-154.
