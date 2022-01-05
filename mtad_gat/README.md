# MTAD-GAT

|||
| :--- | :--- |
| Citekey | ZhaoEtAl2020Multivariate |
| Source Code | own |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python >= 3.7
- torch

## Notes

MTAD-GAT outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for MTAD-GAT
def post_mtad_gat(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 20)
    return ReverseWindowing(window_size=window_size + 1).fit_transform(scores)
```
<!--END:timeeval-post-->

### SR

This algorithm uses Spectral Residuals to clean training data.
We used the code from this [algorithm](../sr).
