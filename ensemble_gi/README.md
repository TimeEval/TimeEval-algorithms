# EnsembleGI

|||
| :--- | :--- |
| Citekey | GaoEtAl2020Ensemble |
| Source | `own` |
| Learning type | unsupervised |
| Input dimensionality | univariate |
|||

## Dependencies

- python 3

## Hyper Parameters

### window_size

The size of the sliding window, in which `w` regions are made discrete.

### ensemble_size

The number of models in the ensemble.

### w_max

The maximum `w` being the spanning length of one discrete sax value.

### a_max

The maximum `a` being the size of the discrete alphabet.

### selectivity

The fraction of models in the ensemble included in the end result.
