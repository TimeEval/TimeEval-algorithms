# Isolation Forest (iForest)

|||
| :--- | :--- |
| Citekey | LiuEtAl2012IsolationBased |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/iforest.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_estimators`: int, optional (default=100)  
  The number of base estimators in the ensemble.

- `max_samples`: int or float, optional (default="auto")  
  The number of samples to draw from X to train each base estimator.
    - If int, then draw `max_samples` samples.
    - If float, then draw `max_samples * X.shape[0]` samples.
    - If "auto", then `max_samples=min(256, n_samples)`.
  If max_samples is larger than the number of samples provided, all samples will be used for all trees (no sampling).

- `max_features`: int or float, optional (default=1.0)  
  The number of features to draw from X to train each base estimator.
    - If int, then draw `max_features` features.
    - If float, then draw `max_features * X.shape[1]` features.

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  Used when fitting to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

- `bootstrap`: bool, optional (default=False)  
  If True, individual trees are fit on random subsets of the training data sampled with replacement.
  If False, sampling without replacement is performed.

- `behaviour`: str, default='old'  
  Behaviour of the `decision_function` which can be either 'old' or 'new'.
  Passing `behaviour='new'` makes the `decision_function` change to match other anomaly detection algorithm API which will be the default behaviour in the future.
  As explained in details in the `offset_` attribute documentation, the `decision_function` becomes dependent on the contamination parameter, in such a way that 0 becomes its natural threshold to detect outliers.
  **REMOVED** (old behavior is used per default!).

- `random_state`: int, RandomState instance or None, optional (default=None)  
  If int, random_state is the seed used by the random number generator;
  If RandomState instance, random_state is the random number generator;
  If None, the random number generator is the RandomState instance used by `np.random`.

- `verbose`: int, optional (default=0)  
  Controls the verbosity of the tree building process.

- `n_jobs`: integer, optional (default=1)  
  The number of jobs to run in parallel for both `fit` and `predict`.
  If -1, then the number of jobs is set to the number of cores.

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
