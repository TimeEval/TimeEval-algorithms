# Series2Graph (S2G)

> **Restricted Access!!**

|||
| :--- | :--- |
| Citekey | BoniolPalpanas2020Series2Graph |
| Source Code | From Paul and Themis |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Original Dependencies

- python=3
- networkx=2.2
- numpy=1.15.4
- scipy=1.1.0
- pandas=0.23.4
- matplotlib=3.0.2
- scikit-learn=0.23.2

## Notes

Series2Graph outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The window size is computed by `(window_size + convolution_size) + query_window_size + 4`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for s2g
def post_s2g(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 50)
    query_window_size = args.get("hyper_params", {}).get("query_window_size", 75)
    convolution_size = args.get("hyper_params", {}).get("convolution_size", window_size // 3)
    size = (window_size + convolution_size) + query_window_size + 4
    return ReverseWindowing(window_size=size).fit_transform(scores)
```
<!--END:timeeval-post-->

## Copyright notice and citation format

> Authors: Paul Boniol, Themis Palpanas, Mohammed Meftah, Emmanuel Remy
> Date: 08/07/2020
> Copyright retained by the authors
> Algorithms protected by patent application FR2005261
> Code provided as is, and can be used only for research purposes
>
> Reference using:
>
> P. Boniol and T. Palpanas, Series2Graph: Graph-based Subsequence Anomaly Detection in Time Series, PVLDB (2020)
>
> P. Boniol and T. Palpanas and M. Meftah and E. Remy, GraphAn: Graph-based Subsequence Anomaly Detection, demo PVLDB (2020)
