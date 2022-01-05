# LaserDBN

|||
| :--- | :--- |
| Citekey | OgbechieEtAl2017Dynamic |
| Source code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | multivariate |
|||

## Dependencies

- python 3
- pomegranate
- networkx
- scikit-learn
- numpy
- pandas

## Notes

The paper used an algorithm call DHC to find the optimal structure of the DBN. However, the algorithm used in the pomegranate library is different. They use a greedy algorithm. The paper was all about video anomaly detection and, therefore, I think the exact implementation is not necessary because we have a different use case anyway. Also, they didn't describe how they used the DHC algorithm. The paper was rather short.
