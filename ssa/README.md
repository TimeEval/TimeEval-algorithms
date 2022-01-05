# SSA

> **Restricted Access!!**

|||
| :--- | :--- |
| Citekey | YaoEtAl2010Online |
| Source Code | https://github.com/johnpaparrizos/AnomalyDetection/tree/master/code/ptsa |
| Learning type | unsupervised |
| Input dimensionality | univariate |

|||

## Notes

SSA works by comparing a reference timeseries to the timeseries that the experiment is being conducted on.
There are 3 main options to run SSA:

- Set `rf_method` to `all`:
- Set `rf_method` to `alpha`:
  You can now set `a` to either a float or a numpy array
  - float: wight that is used to fit the size of the reference and test timeseries
  - nparray: Array of weight used to create a reference TS from the nparray

## Citation

> Yuan Yao, Abhishek Sharma, Leana Golubchik, Ramesh Govindan.
> Online anomaly detection for sensor systems: A simple and efficient approach.
> Performance Evaluation, Volume 67, Issue 11, 2010, Pages 1059-1075, ISSN 0166-5316,
> https://doi.org/10.1016/j.peva.2010.08.018.
