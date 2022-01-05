# DeepAnT

Adapted version of the community implementation of DeepAnT from https://gitlab.hpi.de/akita/bp2020fn1/raw-algorithm-collection/-/tree/master/DeepAnT.

|||
| :--- | :--- |
| Citekey | BasharNayak2020TAnoGAN |
| Source Code | [https://github.com/dev-aadarsh/DeepAnT](https://github.com/dev-aadarsh/DeepAnT) |
| Input Dimensionality | multivariate |
| Learning Type | semi-supervised |
|||

## Dependencies

- python 3
- numpy
- pandas
- pytorch

## Notes

DeepAnT outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The window size is computed by `window_size + prediction_window_size`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for DeepAnT
def _post_deepant(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 45)
    prediction_window_size = args.get("hyper_params", {}).get("prediction_window_size", 1)
    size = window_size + prediction_window_size
    return ReverseWindowing(window_size=size).fit_transform(scores)
```
<!--END:timeeval-post-->
