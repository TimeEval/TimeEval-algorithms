# MultiNormA

> **Restricted Access!!**

|||
| :--- | :--- |
| Citekey | Unpublished |
| Source Code | Leo and Ben, based on code from Paul and Themis |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Original Dependencies

- python==3.6
- numpy==1.15.4
- pandas==0.23.4
- scipy==1.1.0
- tqdm==4.28.1
- stumpy==1.9.2

## Notes

### Input Parameters

The MultiNorma takes a few input parameters that change all runs and a few that change the algorithms behavior or enable
specific steps.

#### Hyper Parameter

- `anomaly_window_size`: subsequence length that is used for searching of recurrent subsequences and distance
  calculation to normal model. Normal Model length is set to three times this parameter. In practice with `timeeval`
  this heuristic worked well: `heuristic:AnomalyLengthHeuristic(agg_type='max')`. Algorithm is quite robust to different
  sizes passed here, only has to be high enough
- `normal_model_percentage`: Percentage of time series which should be used to construct normal model. In practice
  irrelevant, since `max_motifs` parameter overrides this.
- `max_motifs`: Maximum number of recurrent subsequences selected for normal model construction. Highly important
  parameter for resource consumption, if MultiNormA is taking up too much memory or taking too long, try a lower number
  here. Good preset: `4096`
- `random_state`: Random seed for reproducibility.

#### Variants

- `motif_detection`: Specifies how recurrent subsequences for Normal Model construction should be selected from time
  series: `stomp` Matrix Profile based (stomp), `random` Random based (selects `max_motifs / number dimensions` in each
  dimension), `mixed` half of `max_motifs` selected with Matrix Profile based approach and the other half randomly selected.
  Experimentally, the mixed approach works best.
- `sum_dims`: Specifies how dimensions should be handled: `True` sums all dimensions up (not recommended), `False` handles
  them separately (recommended).
- `normalize_join`: `True` applies running mean join normalization after the distance calculation of the time series to the
  normal models before combining the joins dimension wise. `False` disables it. Enabling it is generally recommended.
- `join_combine_method`: Specifies how to combine the joins obtained for all dimensions (distances to normal models)
  . `0` sums them up unweighted (leads to problems in high-dimensions), `1` takes the max of all dimensions, `2` weighs
  the dimensions by a heuristic (1/(std*mean)*range*dimensions), `3` scales up the upper 25% of values, `4`
  exponentiates the join by the number of channels. In our experiments, the max approach (`1`) worked best.

### Post-Processing

The post-processing necessary for NormA is already part of the algorithm, so that for every data point, there is an
anomaly score.

## Copyright notice

The multinormats.lib part is provided under:
> Authors: Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas
> Date: 08/07/2020
> copyright retained by the authors
> algorithms protected by patent application FR2003946
> code provided as is, and can be used only for research purposes

The rest is from:
> Authors: Ben-Noah Engelhaupt, Leo Wendt
> Date: 26/03/2022
