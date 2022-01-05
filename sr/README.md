#SR

|||
| :--- | :--- |
| Citekey | RenEtAl2019TimeSeries |
| Source | `https://github.com/microsoft/anomalydetector` |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Hyper-Parameters

### mag_window

A sliding window average is performed on the time series. This parameter defines its window size.

### score_window

A sliding window average is performed on the time series scores. This parameter defines its window size.

### window_size

Is called batch-size in code but acts like window size. It's the window size of points transformed by `fft`.
