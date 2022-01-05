# Bagel

|||
| :--- | :--- |
| Citekey | LiEtAl2018Robust |
| Source Code | https://github.com/NetManAIOps/Bagel |
| Learning type | semi-supervised |
| Input dimensionality | univariate |
|||

The implementation of 'Robust and Unsupervised KPI Anomaly Detection Based on Conditional Variational Autoencoder'.

## Dependencies

- python >= 3.7
- numpy
- pandas
- scipy
- torch
- sklearn

## Notes

Bagel uses an encoding for timestamps on which the CVAE (Conditional Variational Auto-Encoder) is conditioned on. Hence, a timestamp must be given as a parameter. 
However, it can also handle integers as timestamp values.  
