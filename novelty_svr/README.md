# NoveltySVR

|||
| :--- | :--- |
| Citekey | MaPerkins2003Online |
| Source Code | [https://github.com/fp2556/onlinesvr](https://github.com/fp2556/onlinesvr) |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3
- pandas
- numpy
- scikit-learn
- [pyonlinesvr](https://gitlab.hpi.de/akita/pyonlinesvr)

## Note

- Deviating from the original paper, we do not use a `-eps < residual < +eps)` tolerance interval on the regression residuals to generate occurrences,
  but we use `abs(residual) < 2*eps` for tolerated deviations.
  Residuals that are larger then the tolerance threshold are marked as occurance.
- We use the estimated density of an event as the event anomaly score instead of filtering the events by it to get novel events (3b).
  Afterwards we remove events that have too few occurences according to (3a) (event anomaly score of 0)
  and sum up the event scores for each individual point in the time series to get the anomaly scores.
  The final scores are scaled to `[0; 1]`.
  This makes the threshold parameter `confidence_level` (`c` in the paper) obsolete.
- We don't implement the algorithm variant "Robust Online Novelty Detection" that uses a
  range of different event duration lengths (`n`) and thus requires 3 parameters:
  `n_min`, `n_max`, and `r` (number of durations to consider).
  Since we use event scores and don't perform thresholding on the event density, the
  sensitivity to `n` is reduced.

## Cite as

> Parrelly, Francesco (2007).
> "Online Support Vector Machines for Regression."
> Master thesis. University of Genoa, Italy.
> [PDF](http://onlinesvr.altervista.org/Research/Online%20Support%20Vector%20Regression%20(Parrella%20F.)%20%5B2007%5D.pdf)
