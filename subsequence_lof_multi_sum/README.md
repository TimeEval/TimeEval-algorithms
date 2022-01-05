# Subsequence adaptation of Local outlier factor (LOF) for multivariate datasets

|||
| :--- | :--- |
| Citekey | - |
| Source Code | - |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Notes

This algorithm copies the subsequence-lof algorithm and makes it multivariate-capable by reducing multivariate datasets to a single variable.
For this, dimensions are shifted above zero and summed across all dimensions. For univariate data, it should exactly mirror the subsequence-lof algorithm. 

The results require post-processing, just like the subsequence-lof algorithm.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for sLOF
def post_sLOF(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
