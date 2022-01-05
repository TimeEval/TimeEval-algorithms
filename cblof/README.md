# Cluster-based local outlier factor (CBLOF)

|||
| :--- | :--- |
| Citekey | HeEtAl2003Discovering |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/cblof.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_clusters`: int, optional (default=8)  
  The number of clusters to form as well as the number of centroids to generate.

- `clustering_estimator`: Estimator, optional (default=None)  
  The base clustering algorithm for performing data clustering.
  A valid clustering algorithm should be passed in.
  The estimator should have standard sklearn APIs, fit() and predict().
  The estimator should have attributes ``labels_`` and ``cluster_centers_``.
  If ``cluster_centers_`` is not in the attributes once the model is fit, it is calculated as the mean of the samples in a cluster.
  If not set, CBLOF uses KMeans for scalability.
  See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.
  **REMOVED** default (KMeans) is used!

- `alpha`: float in (0.5, 1), optional (default=0.9)  
  Coefficient for deciding small and large clusters.
  The ratio of the number of samples in large clusters to the number of samples in small clusters.

- `beta`: int or float in (1,), optional (default=5)  
  Coefficient for deciding small and large clusters.
  For a list sorted clusters by size `|C1|, \|C2|, ..., |Cn|, beta = |Ck|/|Ck-1|`.

- `use_weights`: bool, optional (default=False)  
  If set to True, the size of clusters are used as weights in outlier score calculation.

- `check_estimator`: bool, optional (default=False)  
  If set to True, check whether the base estimator is consistent with sklearn standard.
  .. warning::
      check_estimator may throw errors with scikit-learn 0.20 above.
  **REMOVED** must be `False` to work with new scikit-learn version!

- `random_state`: int, RandomState or None, optional (default=None)  
  If int, random_state is the seed used by the random number generator;
  If RandomState instance, random_state is the random number generator;
  If None, the random number generator is the RandomState instance used by `np.random`.

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  When fitting this is used to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

- `n_jobs`: int, optional (default = 1)  
  The number of parallel jobs to run for neighbors search.
  If ``-1``, then the number of jobs is set to the number of CPU cores.
  Affects only kneighbors and kneighbors_graph methods.

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
