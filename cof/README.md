# Local outlier factor (LOF)

|||
| :--- | :--- |
| Citekey | TangEtAl2002Enhancing |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/cof.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_neighbors`: `int`, optional (default=20)  
  Number of neighbors to use by default for k neighbors queries.
  Note that n_neighbors should be less than the number of samples.
  If n_neighbors is larger than the number of samples provided, all samples will be used.

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  When fitting this is used to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
