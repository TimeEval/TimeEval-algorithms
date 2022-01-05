# PhaseSpace-SVM

|||
| :--- | :--- |
| Citekey | MaPerkins2003Timeseries |
| Source Code | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Notes

- We use left-aligned windows to embed the time series into a phase space, because it produces better results.

  Right-aligned embedding:
  ![right](./right_aligned.png)

  Left-aligned embedding:
  ![left](./left_aligned.png)

- We take the raw scores from the SVM as output instead of aggregated binary labels.
  The scores are aggregated using `sum()`.

## Requirements

- numpy
- sklearn
