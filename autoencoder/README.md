# AutoEncoder (AE)

citekey| SakuradaYairi2014Anomaly
:-----:|:-----:
source code| own
Learning type| semi-supervised
input| multivariate

## Notes

* AutoEncoder is trained on all non-anomaly data. Whenever it encounters an anomaly value, the reproduction error is quite higher than the error with non-anomaly instances.
* The test data is put into the AutoEncoder and the scores are returned.
* An assumption is made that all errors are normally distributed with some mean and std. Any error value that follows mean + k*std > threshold or mean - k*std < thereshold is considered as an anomaly. (NOT IMPLEMENTED)
