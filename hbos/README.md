# Histogram-based outlier score (HBOS)

|||
| :--- | :--- |
| Citekey | GoldsteinDengel2012Histogrambased |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/hbos.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_bins`: int, optional (default=10)  
  The number of bins.

- `alpha`: float in (0, 1), optional (default=0.1)  
  The regularizer for preventing overflow.

- `tol`: float in (0, 1), optional (default=0.5)  
  The parameter to decide the flexibility while dealing the samples falling outside the bins.

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  When fitting this is used to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
