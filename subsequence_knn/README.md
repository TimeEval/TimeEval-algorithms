# Sub-KNN

|||
| :--- | :--- |
| Citekey | - |
| Source Code | https://github.com/yzhao062/pyod/blob/master/pyod/models/knn.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `window_size`: `int`, (default=100)
  Size of the sliding windows to extract subsequences as input to KNN.

- `n_neighbors`: `int`, optional (default=5)  
  Number of neighbors to use by default for `kneighbors` queries.

- `algorithm`: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default 'auto')  
  Algorithm used to compute the nearest neighbors:

  - 'ball_tree' will use BallTree
  - 'kd_tree' will use KDTree
  - 'brute' will use a brute-force search.
  - 'auto' will attempt to decide the most appropriate algorithm based on the values passed to `fit` method.

  Note: fitting on sparse input will override the setting of this parameter, using brute force.
  **REMOVED!!**

- `leaf_size`: int, optional (default=30)  
  Leaf size passed to `BallTree`.
  This can affect the speed of the construction and query, as well as the memory required to store the tree.
  The optimal value depends on the nature of the problem.

- `method`: str, optional (default='largest')  

  - 'largest': use the distance to the kth neighbor as the outlier score
  - 'mean': use the average of all k neighbors as the outlier score
  - 'median': use the median of the distance to k neighbors as the outlier score

- `radius`: float, optional (default = 1.0)  
  Range of parameter space to use by default for `radius_neighbors` queries.

- `metric`: string or callable, default 'minkowski'  
  Metric used for the distance computation.
  Any metric from scikit-learn or scipy.spatial.distance can be used.
  If 'precomputed', the training input X is expected to be a distance matrix.
  If metric is a callable function, it is called on each pair of instances (rows) and the resulting value recorded.
  The callable should take two arrays as input and return one value indicating the distance between them.
  This works for Scipy's metrics, but is less efficient than passing the metric name as a string.
  Valid values for metric are:

  - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
  - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']
  
  See the documentation for scipy.spatial.distance for details on these metrics:
  http://docs.scipy.org/doc/scipy/reference/spatial.distance.html.
  **REMOVED!!**

- `p`: integer, optional (default = 2)  
  Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances.
  When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.
  For arbitrary p, minkowski_distance (l_p) is used.
  See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.
  **Renamed to `distance_metric_order`!!**

- `metric_params`: dict, optional (default = None)  
  Additional keyword arguments for the metric function.
  **REMOVED!!**

- `contamination`: float in (0., 0.5), optional (default=0.1)  
  The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
  When fitting this is used to define the threshold on the decision function.
  **Automatically determined by algorithm script!!**

- `n_jobs`: int, optional (default = 1)  
  The number of parallel jobs to run for neighbors search.
  If ``-1``, then the number of jobs is set to the number of CPU cores.
  Affects only kneighbors and kneighbors_graph methods.

## Notes

Sub-KNN outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for subsequence_knn
def post_subsequence_knn(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->


## Citation format (for source code)

> Zhao, Y., Nasrullah, Z. and Li, Z., 2019. PyOD: A Python Toolbox for Scalable Outlier Detection. Journal of machine learning research (JMLR), 20(96), pp.1-7.
