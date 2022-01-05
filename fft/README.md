# Fast Fourier Transform based outlier detection (FFT)

|||
| :--- | :--- |
| Citekey | RasheedEtAl2009Fourier |
| Source Code | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3
- numpy
- (scipy)

## Tweaks

We tweaked the algorithm code a little to fix bugs and make its parametrization more robust/reliable.
The following changes have been done:

- Additional bounds-checks for all of the loops to prevent out-of-bounds exceptions and to speed up processing.
- Specification of the local anomaly threshold in multiples of the standard deviation instead of a fixed score value.
  This is more robust.
- Change (wrt. to the pseudo-code in the paper) of the outlier region calculation from using size bounds on the local outlier indices to the indexes on the original data.
  E.g an outlier region can only have a specific length over the original time series.
