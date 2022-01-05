# ARIMA

> **Restricted Access!!**

|||
| :--- | :--- |
| Citekey | HyndmanAthanasopoulos2018Forecasting |
| Source Code | https://github.com/johnpaparrizos/AnomalyDetection/tree/master/code/ptsa |
| Learning type | unsupervised |
| Input dimensionality | univariate |

|||

## Notes

The ptsa algorithms require sklearn in version 19 to 23. This is checked in the utility.py. Our python image, however, uses a newer sklearn version, which is 24.1 or higher. Hence we removed the check:

```python
if int(sklearn_version.split(".")[1]) < 19: #or int(sklearn_version.split(".")[1]) > 23:
        raise ValueError("Sklearn version error")
```

The ARIMA algorithm performs a fitted check, but that check cannot find the called function in sklearn 20 and higher - the function signature has probably changed. Because the function does not add any computation logic and we do call the fitting in algorithm.py, we removed the checking line from arima.py:

```python
#check_is_fitted(self, ['decision_scores_', 'threshold_', 'labels_'])
```

The sklearn MinMaxScaler has probably moved to a new location in sklearn version 24 or higher. Therefore, we need to import that location in arima.py as well in order to find the MinMaxScaler:

```python
from sklearn.preprocessing import MinMaxScaler
```

## Citation format

> R. Hyndman, Y. Khandakar. Automatic Time Series Forecasting: The forecast Package for R. Journal of Statistical Software, 27(3), 1 - 22, 2008.

See also:
https://otexts.com/fpp2/arima.html
https://research.monash.edu/en/publications/forecasting-principles-and-practice-2
