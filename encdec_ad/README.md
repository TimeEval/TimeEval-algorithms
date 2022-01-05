# EncDec-AD

|||
| :--- | :--- |
| Citekey | MalhotraEtAl2016LSTMbased |
| Source code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pytorch

## Notes

EncDec-AD outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for EncDec-AD
def post_encdec_ad(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    return ReverseWindowing(window_size=2 * window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
