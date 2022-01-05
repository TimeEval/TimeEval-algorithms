# An ensemble detector using multiple random forests on different feature subsets (Random Black Forest)

|||
| :--- | :--- |
| Citekey | - |
| Source Code | own |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Notes

An ensemble windowed multi-output forecasting method using random forest regression and random subspace ensembling (requested by RollsRoyce).
The forecasting error is used as anomaly score.

The regressor ensebmle is trained on a clean time series to look at a fixed window (`train_window_size` points) and predict the next point.
On the test series, the predicted values are compared to the observed ones and the prediction error is returned as anomaly score.
The first `train_window_size` points of the test series don't get an anomaly score (are set to `NaN`), because no predictions are possible for them.
