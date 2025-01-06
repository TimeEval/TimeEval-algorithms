# Half Space Trees (HST)

Half-space trees are an online variant of isolation forests. 
They work well when anomalies are spread out.
However, they do not work well if anomalies are packed together in windows.

|||
| :--- | :--- |
| Citekey | tan2011fast |
| Source Code | https://github.com/online-ml/river/blob/main/river/anomaly/hst.py |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Parameters

- `n_trees`: `int`, optional (default=10)  
  Number of trees to use.
- `height`: `int`, optional (default=8)  
  Height of each tree. A tree of height `h` is made up of `h + 1` levels and
  therefore contains `2 ** (h + 1) - 1` nodes.
- `window_size`: `int`, optional (default=250)
  Number of observations to use for calculating the mass at each node in each tree.

## Citation format (for source code)

 > Tan, S.C., Ting, K.M. and Liu, T.F., 2011, June. Fast anomaly detection for streaming data. In Twenty-Second International Joint Conference on Artificial Intelligence.](https://www.ijcai.org/Proceedings/11/Papers/254.pdf)
