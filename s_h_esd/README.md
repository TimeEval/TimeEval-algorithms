# S-H-ESD (Twitter)

|||
| :--- | :--- |
| Citekey | HochenbaumEtAl2017Automatic |
| Source Code | https://github.com/zrnsm/pyculiarity |
| Learning type | unsupervised |
| Input dimensionality | univariate |

## Citation format (for source code)
> Hochenbaum, J., Vallis, O. S., & Kejariwal, A. (2017). Automatic anomaly detection in the cloud via statistical learning. arXiv preprint arXiv:1704.07706.

## General Information
S-H-ESD is a statistical technique for detecting global as well as local anomalies.
It uses STL time series decomposition to remove seasonality and trends from the data which allows the method to detect
local anomalies that would otherwise be masked by the trend and seasonality components.
In order to increase robustness against time series with a high percentage of anomalies Median and MAD are used for decomposition as they are less sensitive to the anomalies in the data.
On the residuals that are left a ESD test statistic is computed *k* times with *k* being the number of expected anomalous observations in the data. The most extreme value with respect to the test statistic is removed from the dataset.

### Important:
The algorithm computes a pre-selected number of anomalies. In order to provide anomaly scores for all data points, the algorithm was extended as follows:
- Each data point that is not detected by S-H-ESD receives a score of 0
- Data points that got selected as an anomaly receive a score according to the time of their selection, so that a ranking can be obtained (as the algorithm selects more extreme outliers first)

## Custom Parameters
- *max_anomalies*: The expected relative frequency of anomalies in the dataset
