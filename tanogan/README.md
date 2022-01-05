# TAnoGAN

|||
| :--- | :--- |
| Citekey | BasharNayak2020TAnoGAN |
| Source Code | [https://github.com/mdabashar/TAnoGAN](https://github.com/mdabashar/TAnoGAN) |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pytorch
- scikit-learn

## Notes

TAnoGan outputs anomaly scores for windows with strides.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for TAnoGan
def post_tanogan(scores: np.ndarray, args: dict) -> np.ndarray:
    length = args.get("dataset_details").length  # type: ignore
    window_size = args.get("hyper_params", {}).get("window_size", 30)
    scores = np.repeat(scores, repeats=window_size)
    result = np.full(shape=length, fill_value=np.nan)
    result[:scores.shape[0]] = scores
    return result
```
<!--END:timeeval-post-->
