# DAMP
|||
| :--- | :--- |
| Source Code | own                 |
| Learning type | unsupervised      |
| Input dimensionality | multivariate      |
|||

This approach uses the DAMP algorithm to find anomalies. Its functionality is described in this [paper](https://www.cs.ucr.edu/~eamonn/DAMP_long_version.pdf).

## Output Format

The output will be an anomaly score for every subsequence of size *anomaly_window_size*.

## Dependencies

- python 3
- numpy
- pandas
- stumpy

## Notes

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for damp
def post_damp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
