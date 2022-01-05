# HYBRID ISOLATION FOREST (HIF)

|||
| :--- | :--- |
| Citekey | MarteauEtAl2017Hybrid |
| Source Code | https://github.com/pfmarteau/HIF |
| Learning type | supervised |
| Input dimensionality | multivariate |
|||

## Citation format (for source code)

> Marteau, Pierre-François, Saeid Soheily-Khah, and Nicolas Béchet. "Hybrid Isolation Forest-Application to Intrusion Detection." arXiv preprint arXiv:1705.03800 (2017).

## Custom Parameters

- *ntrees*: Number of trees in the HIF
- *sample_size*: Number of samples each tree receives from the input dataset

Because each tree is created by splitting along a random dimension at a random point in the data, adding more trees or increasing the sample size can help improving the confidence in the anomaly scores at the cost of performance.
