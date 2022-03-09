# mSTAMP

|||
| :--- | :--- |
| Citekey | YehEtAl2016Matrix |
| Source Code | stumpy |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

This approach uses the multidimensional matrix profile (mSTAMP). It generates an MP for every dimension and sums them up.

## Output Format

The output will be an anomaly score for every input data point

## Dependencies

- python 3
- numpy
- pandas
- stumpy

## Notes

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for left_stampi
def post_mstamp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
