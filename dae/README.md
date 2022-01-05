# De-Noising-AutoEncoder (AE)

citekey| SakuradaYairi2014Anomaly
:-----:|:-----:
source code| own
Learning type| semi-supervised
input| multivariate

## Notes

* DeNoising AutoEncoder is trained on all noisy non-anomaly data. Whenever it encounters an anomaly value, the reproduction error is quite higher than the error with non-anomaly instances.
* Noise is inserted in randomly selected inputs and turning them to a value of zero. (salt and pepper noise). The De-Noising-AE learns to reproduce the input with noise. The reproduction error is again used to classify between anomalous and non-anomalous data.
* An assumption is made that all errors are normally distributed with some mean and std. Any error value that follows mean + k*std > threshold or mean - k*std < thereshold is considered as an anomaly. The type of noise added is salt and pepper which usually refer to setting some proportion of inputs to zero.
