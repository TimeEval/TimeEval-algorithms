# Donut

|||
| :--- | :--- |
| Citekey | XuEtAl2018Unsupervised |
| Source Code | https://github.com/NetManAIOps/Donut |
| Learning type | semi-supervised |
| Input dimensionality | univariate |
|||

The implementation of 'Unsupervised Anomaly Detection via Variational Auto-Encoder for Seasonal KPIs in Web Applications'.

## Dependencies

- tensorflow==1.*
- git+https://github.com/thu-ml/zhusuan.git
- git+https://github.com/haowen-xu/tfsnippet.git@v0.1.2

## Notes

Early Stopping is applied through the TrainingLoop-Class. However, it uses an own stopping condition.

Donut outputs anomaly window scores.
Therefore, the results require post-processing.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for Donut
def post_donut(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("window_size", 120)
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
