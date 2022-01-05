# Subsequence Fast-MCD

|||
| :--- | :--- |
| Citekey | - |
| Source | `own` |
| Learning type | semi-supervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3
- sklearn

## Notes

- We first split the univariate timeseries into smaller subsequences using sliding windows.
- Each window is then a multidimensional object fed into the Fast-MCD algorithm.
- Afterward, Fast-MCD works on the subsequences.

Subsequence Fast-MCD therefore outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for Subsequence Fast-MCD
def post_sfmcd(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
