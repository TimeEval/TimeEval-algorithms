# TSBitmap

|||
| :--- | :--- |
| Citekey | WeiEtAl2005Assumptionfree |
| Source Code | https://github.com/binhmop/tsbitmaps |
| Learning type | unsupervised |
| Input dimensionality | univariate |

|||

## Citation format (for source code)

> Wei, L., Kumar, N., Lolla, V. N., Keogh, E. J., Lonardi, S., & Chotirat (Ann) Ratanamahatana. (2005, June). Assumption-Free Anomaly Detection in Time Series. In SSDBM (Vol. 5, pp. 237-242).

## General Notes

The algorithm first discretizes the timeseries by applying the SAX method to it. It converts the numerical data points to characters of a string by binning it on a window-basis. It also reduces the dimensionality along the time axis with a PAA (Piecewise Aggregate Approximation). Thus, each symbol represents a time segment.

Then, we slide through the discretized time series and at each time segment obtain two bitmaps: one that looks in the future and one that looks in the past. These are then compared to calculate an anomaly score.

After termination, we need to decompress the scores so that each score (who now represents a time segment) gets mapped to its original timestamp.

## Custom Parameters

- *feature_window_size*: The scope where the discretization is applied - one sax word per feature window
- *bins*: Number of bins (or alphabet size) for sax discretization
- *level_size*: Length of the subwords which frequencies are to be counted
- *lead_window_size*: Tells the algorithm how far to look in the future (lead bitmap calculation)
- *lag_window_size*: Tells the algorithm how far to look back into the past (lag bitmap calculation)
- *compression_ratio*: Specifies how many consecutive timestamps get collapsed into one by the PAA
