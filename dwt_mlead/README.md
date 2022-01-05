# Discrete Wavelet Transform and Maximimum Likelihood Estimation for Anomaly Detection in time series (DWT-MLEAD)

|||
| :--- | :--- |
| Citekey | ThillEtAl2017Time |
| Source Code | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Notes

- In contrast to the paper, we use window sizes for the coefficients that decrease with the level number.
  This makes more sense, because otherwise we would have too few items to slide the window over.
- We exclude the highest level coefficients, because they contain only a single entry, where we cannot slide a window of length 2 over.
- We have **not implemented monte carlo sampling** for the quantile estimation.

## Requirements

- numpy
- sklearn
- pywavelets
