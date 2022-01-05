# Probabilistic Suffix Tree (PST)

|||
| :--- | :--- |
| Citekey | SunEtAl2006Mining |
| Source code | [http://r-forge.r-project.org/projects/pst](http://r-forge.r-project.org/projects/pst) |
| Learning type | unsupervised |
| Input Dimensionality | univariate |
|||

## Dependencies

- System dependencies (`apt`)
  - build-essential (make, gcc, ...)
  - r-base
- R-packages
  - jsonlite
  - PST
  - TraMineR
  - arules
  - pkgcond
  - BBmisc

## Notes

In the paper [Mining for Outliers in Sequential Databases](https://doi.org/10.1137/1.9781611972764.9) using a PST for anomaly detection is only proposed for discrete data.
Since we want to evaluate this algorithm also using continuous data, we have added a **discretization step** before the actual algorithm.
This discretization step discretizes the input time-series into breaks number of buckets by frequency (breaks is a custom parameter which has to be given to the algorithm).

PST computes anomaly scores for sequences.
However, this algorithm already converts those anomaly scores for sequences into anomaly scores for points by computing the average anomaly score for each point over all sequences it is included.
