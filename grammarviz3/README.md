# GrammarViz 3

|||
| :--- | :--- |
| Citekey | SeninEtAl2015Time |
| Source Code | [https://github.com/GrammarViz2/grammarviz2_src](https://github.com/GrammarViz2/grammarviz2_src) |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Notes

GrammarViz outputs the distance of discords to their nearest non-self match and their length.
Therefore, the results require post-processing.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
import pandas as pd
import numpy as np

from scipy.sparse import csc_matrix, hstack
from timeeval.utils.window import ReverseWindowing
from timeeval import AlgorithmParameter

# post-processing for GrammarViz
def post_grammarviz(algorithm_parameter: AlgorithmParameter, args: dict) -> np.ndarray:
    if isinstance(algorithm_parameter, np.ndarray):
        results = pd.DataFrame(algorithm_parameter, columns=["index", "score", "length"])
        results = results.set_index("index")
    else:
        results = pd.read_csv(algorithm_parameter, header=None, index_col=0, names=["index", "score", "length"])
    anomalies = results[results["score"] > .0]

    # use scipy sparse matrix to save memory
    matrix = csc_matrix((len(results), 1), dtype=np.float64)
    counts = np.zeros(len(results))
    for i, row in anomalies.iterrows():
        idx = int(row.name)
        length = int(row["length"])
        tmp = np.zeros(len(results))
        tmp[idx:idx + length] = np.repeat([row["score"]], repeats=length)
        tmp = tmp.reshape(-1, 1)
        matrix = hstack([matrix, tmp])
        counts[idx:idx + length] += 1
    sums = matrix.sum(axis=1)
    counts = counts.reshape(-1, 1)
    scores = np.zeros_like(sums)
    np.divide(sums, counts, out=scores, where=counts != 0)
    # returns the completely flattened array (from `[[1.2], [2.3]]` to `[1.2, 2.3]`)
    return scores.A1
```
<!--END:timeeval-post-->

## Citation format

> Pavel Senin, Jessica Lin, Xing Wang, Tim Oates, Sunil Gandhi, Arnold P. Boedihardjo, Crystal Chen, and Susan Frankenstein. 2018. GrammarViz 3.0: Interactive Discovery of Variable-Length Time Series Patterns. ACM Trans. Knowl. Discov. Data 12, 1, Article 10 (February 2018), 28 pages. DOI: https://doi.org/10.1145/3051126
>
> Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., Lerner, M., Time series anomaly discovery with grammar-based compression, The International Conference on Extending Database Technology, EDBT 15.
