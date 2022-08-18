# Multivariate subsequence adaptation of local outlier factor (Multi-Sub-LOF)

|||
| :--- | :--- |
| Citekey | - |
| Source Code | - |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Notes

This algorithm copies the Sub-LOF algorithm and makes it multivariate-capable by either concatenating the channels or reducing the channels to a single dimension.
For this, dimensions are shifted above zero and summed across all dimensions. For univariate data, it should exactly mirror the subsequence-lof algorithm. 


This algorithm outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for multi-sub-LOF
def post_multi_subsequence_lof(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
