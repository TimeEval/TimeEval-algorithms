# TripleES

|||
| :--- | :--- |
| Citekey | Aboode2018Anomaly |
| Source Code | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |

## Citation format (for source code)

> Aboode, A. (2018). Anomaly detection in time series data based on holt-winters method.

## General Notes

Holt Winter's method (also known as triple exponential smoothing) models the time series by assuming it is composed
of a trend and seasonality component, which are added to the classic exponential smoothing estimation used for forecasting.
The anomaly detection method described in the paper additionally assumes the residuals of the forecast follow
a standard normal distribution. It is implemented as a triple exp. smoothing model that is fit to a sliding window
of data. The window is used to forecast the next point. The residual of the forecast divided by the standard deviation
of the residuals inside the window gives an anomaly score.

## Custom parameters

The 'seasonal_periods' parameters refers to the number of time steps in the data at which periodic/seasonal events occur. Examples could be 7 days in daily data, 4 quarters in quarterly data, 12 hours in hourly data, etc.
