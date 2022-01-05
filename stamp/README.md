# STAMP

|||
| :--- | :--- |
| Citekey | YehEtAl2016Matrix |
| Source Code | [https://github.com/matrix-profile-foundation/tsmp](https://github.com/matrix-profile-foundation/tsmp) |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Original Dependencies

- System dependencies (`apt`)
  - build-essential (make, gcc, ...)
  - r-base
- R-packages
  - tsmp

## Notes

STAMP outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The output window size is equal to `window_size`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for stamp
def post_stamp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    if window_size < 4:
      print("WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4")
      window_size = 4
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->

## Description: Anytime univariate STAMP algorithm (parallel version)

Computes the best so far Matrix Profile and Profile Index for Univariate Time Series.

### Details

The Matrix Profile, has the potential to revolutionize time series data mining because of its generality, versatility, simplicity, and scalability.
In particular, it has implications for time series motif discovery, time series joins, shapelet discovery (classification), density
estimation, semantic segmentation, visualization, rule discovery, clustering, etc.
The anytime STAMP computes the Matrix Profile and Profile Index in such a manner that it can be stopped before its complete calculation and return the best so far results allowing ultra-fast approximate solutions.
`verbose` changes how much information is printed by this function; `0` means nothing, `1` means text, `2` adds the progress bar, `3` adds the finish sound.
`exclusion_zone` is used to avoid trivial matches; if query data is provided (join similarity), this parameter is ignored.

### Params

- `window_size` (`int`):
  Size of the sliding window.
- `exclusion_zone` (`numeric`):
  Size of the exclusion zone, based on window size.
  See details.
  (Default is `1/2`).
- `verbose` (`int`):
  See details.
  (Default is `1`).
- `s_size` (`numeric`):
  For anytime algorithm, represents the size (in observations) the random calculation will occur.
  (Default is `Inf`).
- `weight` (`vector` of `numeric` with the same length of the `window_size` or `NULL`):
  This is a MASS extension to weigh the query.
  **Not needed for self-join: REMOVED!**
- `n_jobs` (`int`):
  Number of workers for parallel.
  (Default is `1`).

Returns a `MatrixProfile` object, a `list` with the matrix profile `mp`, profile index `pi` left and right matrix profile `lmp`, `rmp` and profile index `lpi`, `rpi`, window size `w` and exclusion zone `ez`.

### Example

```R
mp <- stamp_par(mp_toy_data$data[1:200, 1], window_size = 30, verbose = 0)
ref_data <- mp_toy_data$data[, 1]
query_data <- mp_toy_data$data[, 2]
# self similarity
mp <- stamp(ref_data, window_size = 30, s_size = round(nrow(ref_data) * 0.1))
```

## Citation and Reference

> - Yeh CCM, Zhu Y, Ulanova L, Begum N, Ding Y, Dau HA, et al. Matrix profile I: All pairs similarity joins for time series: A unifying view that includes motifs, discords and shapelets. Proc - IEEE Int Conf Data Mining, ICDM. 2017;1317-22.
> - Zhu Y, Imamura M, Nikovski D, Keogh E. Matrix Profile VII: Time Series Chains: A New Primitive for Time Series Data Mining. Knowl Inf Syst. 2018 Jun 2;1-27.
>
> Website: <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>
