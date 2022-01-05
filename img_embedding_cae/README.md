# ImageEmbeddingCAE

|||
| :--- | :--- |
| Citekey | GarciaEtAl2020Time |
| Source Code | `own` |
| Learning type | semi-supervised |
| Input dimensionality | univariate |

## Citation format (for source code)

> Garcia, G. R., Michau, G., Ducoffe, M., Gupta, J. S., & Fink, O. (2021). Temporal signals to images: Monitoring the condition of industrial assets with deep learning image processing algorithms. Proceedings of the Institution of Mechanical Engineers, Part O: Journal of Risk and Reliability, 1748006X21994446.

## General Notes

This method splits the input one-dimensional time series into tumbling windows and encodes each window as an image.
For this, a continuous wavelet transform (CWT) is applied.
The resulting frequencies and coefficients can be interpreted as a heatmap (scalogram).
A healthy dataset is used to capture relevant patterns found in a healthy environment.
If the trained convolutional autoencoder then receives anomalous time series windows its ability to reconstruct the window is expected to be low. The residuals (l1-Norm) are computed for each window and output as an anomaly score.

The window anomaly scores are automatically converted to point anomaly scores within the method.

## Custom Parameters
- _anomaly_window_size_

This parameter describes how many data points along the time axis are converted into an image.
The time series is split into tumbling windows of this size and then downscaled by a constant factor of 8.
Example: $$\frac{512}{8} = 64$$
Meaning each image would be of size 64 times 64.

- _kernel_size_

The kernel size specifies the size of the convolution kernels in both directions. As a stride, the size of the kernel itself is used to obtain tumbling windows.
This requires the images as well as the convolution output features to be divisible by the kernel size.
Example: $$\frac{1000}{8} = 125; \frac{125}{5} = 25; \frac{25}{5} = 5$$

- _num_kernels_

The number of kernels used in each convolution and deconvolution layer.
Can be chosen freely, but more kernels take more time to train.

- _latent_size_ 

The number of neurons in the bottleneck layer of the autoencoder. Can be chosen freely.

### Parameter correction
To ensure valid input parameters that fulfill the constrains presented for the kernel size and the input images,
a parameter correction is used that corrects the _anomaly_window_size_ to the next larger valid size that fulfills the constraints. It is chosen by the following equation: $$floor(\frac{A}{8K^2} + 1) 8K^2$$
with A being the _anomaly_window_size_ and K being the _kernel_size_.