# DeepNAP

|||
| :--- | :--- |
| Citekey | KimEtAl2018DeepNAP |
| Source Code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pytorch

## Notes

DeepNAP outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for DeepNAP
def post_deepnap(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 15)
    return ReverseWindowing(window_size=window_size * 2).fit_transform(scores)
```
<!--END:timeeval-post-->
