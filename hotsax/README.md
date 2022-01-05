# HOT-SAX

|||
| :--- | :--- |
| Citekey | KeoghEtAl2005HOT |
| Source Code | https://github.com/seninp/saxpy |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Notes

We use the source code from the Github repository instead of [the published library on pypi](https://pypi.org/project/saxpy/) because it contains runtime improvements and edge case fixes.

HOT-SAX outputs the distance of discords to their nearest non-self match.
Therefore, the results require post-processing.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix, hstack

from timeeval.utils.window import ReverseWindowing
from timeeval import AlgorithmParameter

# post-processing for HOT-SAX
def post_hotsax(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 100)
    if isinstance(algorithm_parameter, np.ndarray):
        results = pd.DataFrame(algorithm_parameter)
    else:
        results = pd.read_csv(algorithm_parameter)
    results.columns = ["score"]
    anomalies = results[results["score"] > .0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        tmp = np.zeros(len(results))
        tmp[idx:idx + window_size] = np.repeat([row["score"]], repeats=window_size)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx:idx + window_size] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1
```
<!--END:timeeval-post-->

## Citation format

> Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., Lerner, M., GrammarViz 2.0: a tool for grammar-based pattern discovery in time series, ECML/PKDD Conference, 2014.
