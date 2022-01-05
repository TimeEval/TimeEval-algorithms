# Hybrid-KNN

|||
| :--- | :--- |
| Citekey | SongEtAl2017Hybrid |
| Source code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pytorch

## Notes

Hybrid-KNN outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for Hybrid-KNN
def post_hybrid_knn(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
