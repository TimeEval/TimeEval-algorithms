# TARZAN

|||
| :--- | :--- |
| Citekey | KeoghEtAl2002Finding |
| Source | `community` |
| Learning type | semi-supervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3

## Notes

TARZAN outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for TARZAN
def post_tarzan(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
