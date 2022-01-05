# MSCRED

|||
| :--- | :--- |
| Citekey | ZhangEtAl2019Deep |
| Source Code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pytorch

## Notes

MSCRED outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
import numpy as np
from timeeval.utils.window import ReverseWindowing
# post-processing for MSCRED
def post_mscred(scores: np.ndarray, args: dict) -> np.ndarray:
    ds_length = args.get("dataset_details").length  # type: ignore
    gap_time = args.get("hyper_params", {}).get("gap_time", 10)
    window_size = args.get("hyper_params", {}).get("window_size", 5)
    max_window_size = max(args.get("hyper_params", {}).get("windows", [10, 30, 60]))
    offset = (ds_length - (max_window_size - 1)) % gap_time
    image_scores = ReverseWindowing(window_size=window_size).fit_transform(scores)
    return np.concatenate([np.repeat(image_scores[:-offset], gap_time), image_scores[-offset:]])
```
<!--END:timeeval-post-->
