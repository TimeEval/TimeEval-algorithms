# Subsequence adaptation of Local outlier factor (LOF)

|||
| :--- | :--- |
| Citekey | - |
| Source Code | - |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Notes

Subsequence LOF outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for sLOF
def post_sLOF(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
