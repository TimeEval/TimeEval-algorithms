# Left STAMPi

|||
| :--- | :--- |
| Citekey | YehEtAl2016Matrix |
| Source Code | stumpy |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

This approach uses the left matrix profile (LMP). That is extended with every new point in the stream. 
The already existing LMP entries are not updated.

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
def post_left_stampi(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 50)
    n_init_train = args.get("hyper_params", {}).get("n_init_train", 50)
    if window_size > n_init_train:
        print(f"WARN: anomaly_window_size is larger than n_init_train. Dynamically fixing it by setting anomaly_window_size to n_init_train={n_init_train}")
        window_size = n_init_train
    if window_size < 3:
        print("WARN: anomaly_window_size must be at least 3. Dynamically fixing it by setting anomaly_window_size to 3")
        window_size = 3
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->
