# Principle Component Classifier (PCC)

|||
| :--- | :--- |
| Citekey | ShyuEtAl2003novel |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/pca.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_components`: int, float, None or string  
  Number of components to keep.
  If `n_components` is not set all components are kept: `n_components == min(n_samples, n_features)`.
  If `n_components == 'mle'` and `svd_solver == 'full'`, Minka's MLE is usedto guess the dimension.
  If `0 < n_components < 1` and `svd_solver == 'full'`, select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by `n_components`.
  `n_components` cannot be equal to `n_features` for `svd_solver == 'arpack'`.

- `n_selected_components`: int, optional (default=None)  
  Number of selected principal components for calculating the outlier scores.
  It is not necessarily equal to the total number of the principal components.
  If not set, use all principal components.

- `whiten`: bool, optional (default False)  
  When True the `components_` vectors are multiplied by the square root of `n_samples` and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances.
  Whitening will remove some information from the transformed signal (the relative variance scales of the components) but can sometime improve the predictive accuracy of the downstream estimators by making their data respect some hard-wired assumptions.

- `svd_solver`: string {'auto', 'full', 'arpack', 'randomized'} (default 'auto')  

  - 'auto': the solver is selected by a default policy based on `X.shape` and `n_components`:
    if the input data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest dimension of the data, then the more efficient 'randomized' method is enabled.
    Otherwise the exact full SVD is computed and optionally truncated afterwards.
  - 'full': run exact full SVD calling the standard LAPACK solver via `scipy.linalg.svd` and select the components by postprocessing.
  - 'arpack' : run SVD truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds`.
    It requires strictly `0 < n_components < X.shape[1]`.
  - 'randomized' : run randomized SVD by the method of Halko et al.

- `tol`: float >= 0, optional (default .0)  
  Tolerance for singular values computed by `svd_solver == 'arpack'`.

- `iterated_power`: int >= 0, or 'auto', (default 'auto')  
  Number of iterations for the power method computed by `svd_solver == 'randomized'`.

- `random_state`: int, RandomState instance or None, optional (default None)  
  If int, random_state is the seed used by the random number generator;
  If RandomState instance, random_state is the random number generator;
  If None, the random number generator is the RandomState instance used by `np.random`.
  Used when `svd_solver == 'arpack' or svd_solver == 'randomized'`.

- `weighted`: bool, optional (default=True)  
  If True, the eigenvalues are used in score computation.
  The eigenvectors with small eigenvalues comes with more importance in outlier score calculation.
  **REMOVED** paper uses this!

- `copy`: bool (default True)  
  If False, data passed to fit are overwritten and running `fit(X).transform(X)` will not yield the expected results, use `fit_transform(X)` instead.
  **REMOVED** always copy!

- `standardization`: bool, optional (default=True)  
  If True, perform standardization first to convert data to zero mean and unit variance.
  See http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html.
  **REMOVED** always standardize!

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  Used when fitting to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
