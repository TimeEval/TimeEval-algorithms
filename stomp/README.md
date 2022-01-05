# STOMP

|||
| :--- | :--- |
| Citekey | ZhuEtAl2016Matrix |
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

STOMP outputs anomaly scores for windows.
The results require post-processing.
The scores for each point can be assigned by aggregating the anomaly scores for each window the point is included in.
The output window size is equal to `window_size`.

U can use the following code snippet for the post-processing step in TimeEval (default parameters directly filled in from the source code):

<!--BEGIN:timeeval-post-->
```python
from timeeval.utils.window import ReverseWindowing
# post-processing for stomp
def post_stomp(scores: np.ndarray, args: dict) -> np.ndarray:
    window_size = args.get("hyper_params", {}).get("anomaly_window_size", 30)
    if window_size < 4:
      print("WARN: window_size must be at least 4. Dynamically fixing it by setting window_size to 4")
      window_size = 4
    return ReverseWindowing(window_size=window_size).fit_transform(scores)
```
<!--END:timeeval-post-->

## Description: Univariate STOMP algorithm (parallel version)

Computes the Matrix Profile and Profile Index for Univariate Time Series.

### Details

The Matrix Profile, has the potential to revolutionize time series data mining because of its generality, versatility, simplicity and scalability.
In particular it has implications for time series motif discovery, time series joins, shapelet discovery (classification), density estimation, semantic segmentation, visualization, rule discovery, clustering etc.

`verbose` changes how much information is printed by this function;
`0` means nothing, `1` means text, `2` adds the progress bar, `3` adds the finish sound.
`exclusion_zone` is used to avoid  trivial matches;
if a query data is provided (join similarity), this parameter is ignored.

### Parameters

- `window_size` (`int`):
  Size of the sliding window.
- `exclusion_zone` (`numeric`):
  Size of the exclusion zone, based on window size.
  See details.
  (Default is `1/2`).
- `verbose` (`int`):
  See details.
  (Default is `1`).
- `n_jobs` (`int`):
  Number of workers for parallel.
  (Default is `1`).

Returns a `MatrixProfile` object, a `list` with the matrix profile `mp`, profile index `pi` left and right matrix profile `lmp`, `rmp` and profile index `lpi`, `rpi`, window size `w` and exclusion zone `ez`.

### Example

```R
# using threads
mp <- stomp_par(mp_toy_data$data[1:400, 1], window_size = 30, verbose = 0)
```

## Citation and Reference

> Zhu Y, Zimmerman Z, Senobari NS, Yeh CM, Funning G. Matrix Profile II : Exploiting a Novel Algorithm and GPUs to Break the One Hundred Million Barrier for Time Series Motifs and Joins. Icdm. 2016 Jan 22;54(1):739-48.
>
> Website: <http://www.cs.ucr.edu/~eamonn/MatrixProfile.html>
