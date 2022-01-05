# Seasonal Auto-Regressive Integrated Moving Average (SARIMA)

|||
| :--- | :--- |
| Citekey | GreisEtAl2018Comparing |
| Source Code | https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html#statsmodels.tsa.statespace.sarimax.SARIMAX |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3
- pandas
- numpy
- statsmodels
- pmdarima

## Notes

- The (p,d,q,P,D,Q) orders of the SARIMA model are automatically determined using statistical tests and stepwise refinement (grid search).
  You can overwrite this tuning behavior by supplying your orders to `fixed_orders`, e.g. `fixed_orders = { "order": (2, 0, 3), "seasonal_order": (0, 0, 2) }`.
  The period `m` is automatically added.
- Using `exhaustive_search=True`, the orders are searched for using a grid search without any prior statistical tests.
  This drastically increases runtime, but finds the optimal model.
- The point anomaly score is the absolute error between forecast and original value.
- We use SARIMA in an iterative way, fitting model on the first `train_window_size` points, forecasting `forecast_window_size` points, and re-calibrating the SARIMA-parameters after each prediction.
- If `max_lag` is set, then the order of the SARIMA model is retrained after `max_lag` points before making further predictions.
