# Torsk

|||
| :--- | :--- |
| Citekey | HeimAvery2019Adaptive |
| Source Code | [https://github.com/nmheim/torsk](https://github.com/nmheim/torsk) |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- joblib
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn

## Notes

- Returns window anomaly scores.
  We can transform them back to point scores using post-processing:

<!--BEGIN:timeeval-post-->
  ```python
  from timeeval.utils.window import ReverseWindowing
  # post-processing for Torsk
  def _post_torsk(scores: np.ndarray, args: dict) -> np.ndarray:
      pred_size = args.get("hyper_params", {}).get("prediction_window_size", 20)
      context_window_size = args.get("hyper_params", {}).get("context_window_size", 10)
      size = pred_size * context_window_size + 1
      return ReverseWindowing(window_size=size).fit_transform(scores)
  ```
<!--END:timeeval-post-->

- We don't use the input mapping feature to its full potential.
  Only a single random weight mapping is used.
  The parameters of this mapping are exposed.

- The pytorch implementation had various errors, so we disabled this backend and removed the code for it.

- Information about the different window sizes and why the actual window size is one larger than `train_size + pred_size`:

  ```plain
  dataset_size = 40
  train_size = 20
  transient_size = 10
  pred_size  = 10
  steps = dataset_size - train_size - 1 - pred_size = 9

  ------------------------------------------
  |          dataset size = 40             |
  ------------------------------------------

  For training:
  ++++++++++++++++++++++++++++++++++
  |         window size = 31       |
  |       train 21      |  pred 10 |
  |     train_x 20     |    -      |
  ||      train_y 20    |    -     |
  ++++++++++++++++++++++++++++++++++

  For optimization:
  ++++++++++++++++++++++++++++++++++
  |         window size = 31       |
  |  tran 10 | opt data |     -    |
  ++++++++++++++++++++++++++++++++++

  For prediction:
  ++++++++++++++++++++++++++++++++++
  |         window size = 31       |
  |          -          |  pred 10 |
  ++++++++++++++++++++++++++++++++++
  ```

- Parameter values are very dependent on a dataset.
  The default parameters are just rough estimates.
