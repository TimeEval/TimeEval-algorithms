# mVALMOD

|||
| :--- |:---|
| Citekey | LinardiEtAl2020Matrix |
| Source Code | [https://github.com/matrix-profile-foundation/tsmp](https://github.com/matrix-profile-foundation/tsmp) |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Original Dependencies

- System dependencies (`apt`)
  - build-essential (make, gcc, ...)
  - r-base
- R-packages
  - tsmp

## Notes

The mVALMOD algorithm is an experimental extension to VALMOD. With this version we run the original VALMOD algorithm on every channel of a multivariate time series.
The matrix profiles for each channel are then added together to get a univariate matrix profile that can be used as an anomaly score.

VALMOD outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The output window size is equal to `min_anomaly_window_size`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for valmod
def post_valmod(scores: np.ndarray, args: dict) -> np.ndarray:
    window_min = args.get("hyper_params", {}).get("min_anomaly_window_size", 30)
    window_min = max(window_min, 4)
    return ReverseWindowing(window_size=window_min).fit_transform(scores)
```
<!--END:timeeval-post-->

## Description: Variable Length Motif Discovery

Computes the Matrix Profile and Profile Index for a range of query window sizes.

### Details

This algorithm uses an exact algorithm based on a novel lower bounding technique, which is specifically designed for the motif discovery problem.
`verbose` changes how much information is printed by this function;
`0` means nothing, `1` means text, `2` adds the progress bar, `3` adds the finish sound.
`exclusion_zone` is used to avoid  trivial matches;
if a query data is provided (join similarity), this parameter is ignored.

Paper that implements `skimp()` suggests that window_max / window_min > 1.24 begins to weakening pruning in `valmod()`.

### Parameters

- `window_min` (`int`):
  Minimum size of the sliding window.
- `window_max` (`int`):
  Maximum size of the sliding window.
- `heap_size` (`int`):
  Size of the distance profile heap buffer.
  (Default is `50`).
- `exclusion_zone` (`numeric`):
  Size of the exclusion zone, based on window size.
  See details.
  (Default is `1/2`).
- `lb` (`logical`):
  If `FALSE` all window sizes will be calculated using STOMP instead of pruning.
  This is just for academic purposes.
  (Default is `TRUE`).
  **REMOVED!**
- `verbose` (`int`):
  See details.
  (Default is `1`).

Returns a `Valmod` object, a `list` with the matrix profile `mp`, profile index `pi` left and right matrix profile `lmp`, `rmp` and profile index `lpi`, `rpi`, best window size `w` for each index and exclusion zone `ez`.
Additionally: `evolution_motif` the best motif distance per window size, and non-length normalized versions of `mp`, `pi` and `w`:
`mpnn`, `pinn` and `wnn`.

### Example

```R
mp <- valmod(mp_toy_data$data[1:200, 1], window_min = 30, window_max = 40, verbose = 0)
\donttest{
    ref_data <- mp_toy_data$data[, 1]
    query_data <- mp_toy_data$data[, 2]
    # self similarity
    mp <- valmod(ref_data, window_min = 30, window_max = 40)
    # join similarity
    mp <- valmod(ref_data, query_data, window_min = 30, window_max = 40)
}
```

## Citation and Reference

> Linardi M, Zhu Y, Palpanas T, Keogh E. VALMOD: A Suite for Easy and Exact Detection of Variable Length Motifs in Data Series. In: Proceedings of the 2018 International Conference on Management of Data - SIGMOD '18. New York, New York, USA: ACM Press; 2018. p. 1757-60.
>
> Website: <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>
