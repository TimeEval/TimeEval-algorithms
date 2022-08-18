# MultiGrammarViz

|||
| :--- | :--- |
| Citekey | - |
| Source Code | own |
| Learning type | unsupervised |
| Input dimensionality | multivariate |
|||

## Notes

This is a modified version of the grammarviz algorithm. Modifications include the possibility to classify multivariate time series, and general quality-of-life additions.
Furthermore, parameters for additional configuration of output algorithms were added, and the need for post-processing was removed.
The most important parameters are `output_mode` and `multi_strategy`.
`output_mode` specifies the algorithm which will generate the anomaly scores and
`multi_strategy` specifies which adaption to the multivariate case should be used.
This only applies for time series with more than one dimension.
The univariate implementation uses `output_mode` of `2`.

| Output mode value | algorithm |
| :---------------: | :-------- |
| 0 | rule density                |
| 1 | discord discovery (RRA)     |
| 2 | modified brute-force HOTSAX |

| Multivariate strategy value | algorithm |
| :-------------------------: | :-------- |
| 0 | merge all dimensions        |
| 1 | merge correlated dimensions |
| 2 | merge no dimensions         |

## Citation format

> Pavel Senin, Jessica Lin, Xing Wang, Tim Oates, Sunil Gandhi, Arnold P. Boedihardjo, Crystal Chen, and Susan Frankenstein. 2018. GrammarViz 3.0: Interactive Discovery of Variable-Length Time Series Patterns. ACM Trans. Knowl. Discov. Data 12, 1, Article 10 (February 2018), 28 pages. DOI: <https://doi.org/10.1145/3051126>
>
> Senin, P., Lin, J., Wang, X., Oates, T., Gandhi, S., Boedihardjo, A.P., Chen, C., Frankenstein, S., Lerner, M., Time series anomaly discovery with grammar-based compression, The International Conference on Extending Database Technology, EDBT 15.
