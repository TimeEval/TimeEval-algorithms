# Copula-based outlier detector (COPOD)

|||
| :--- | :--- |
| Citekey | LiEtAl2020COPOD |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/copod.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  When fitting this is used to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
