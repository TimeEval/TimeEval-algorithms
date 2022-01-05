# NormA

> **Restricted Access!!**

|||
| :--- | :--- |
| Citekey | BoniolEtAl2021Unsupervised |
| Source Code | From Paul and Themis |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Original Dependencies

- python==3.6
- numpy==1.15.4
- pandas==0.23.4
- scipy==1.1.0
- tqdm==4.28.1
- tslearn==0.1.29 (requires cython)
- bundled dependency:
  - matrix_profile==0.1

## Notes

NormA outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The window size is computed by `2 * (anomaly_window_size - 1) + 1`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for norma
def _post_norma(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 20)
    size = 2 * window_size - 1
    return ReverseWindowing(window_size=size).fit_transform(scores)
```
<!--END:timeeval-post-->

## Copyright notice

> Authors: Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas
> Date: 08/07/2020
> copyright retained by the authors
> algorithms protected by patent application FR2003946
> code provided as is, and can be used only for research purposes
