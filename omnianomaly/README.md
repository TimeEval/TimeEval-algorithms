# OmniAnomaly

|||
| :--- | :--- |
| Citekey | SuEtAl2019Robust |
| Source Code | [https://github.com/NetManAIOps/OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)|
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python=3
- see requirements.txt

## License

[link to license](./LICENSE)

## Notes

OmniAnomaly outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.

You can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for OmniAnomaly
def post_omni_anomaly(scores: np.ndarray, args: dict) -> np.ndarray:
    window_length = args.get("hyper_params", {}).get("window_size", 100)
    return ReverseWindowing(window_size=window_length).fit_transform(scores)
```
<!--END:timeeval-post-->

## Further Notes

Early Stopping is applied through the TrainingLoop-Class. However, it uses an own stopping condition.
