<div align="center">
<img width="130px" src="./timeeval-algorithms.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval Algorithms</h1>
<p>
  Time Series Anomaly Detection Algorithms for TimeEval.
</p>
</div>

## Description

This repository contains a collection of containerized (dockerized) time series anomaly detection methods that can easily be evaluated using [TimeEval](https://github.com/HPI-Information-Systems/TimeEval).
Some of the algorithm's source code is access restricted and we just provide the TimeEval stubs and manifests. We are happy to share our TimeEval adaptations of excluded algorithms upon request, if the original authors approve this.

Each folder contains the implementation of an algorithm that will be build into a runnable Docker container using CI.
The namespace prefix (repository) for the built Docker images is `registry.gitlab.hpi.de/akita/i/`.

## Overview

| Algorithm (folder) | Image | Language | Base image | Learning Type | Input Dimensionality |
| :----------------- | :---- | :------- | :--------- | :------------ | :------------------- |
| [arima](./arima) (_restricted access_) | `registry.gitlab.hpi.de/akita/i/arima` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [autoencoder](./autoencoder) | `registry.gitlab.hpi.de/akita/i/autoencoder` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [bagel](./bagel) | `registry.gitlab.hpi.de/akita/i/bagel` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [baseline_increasing](./baseline_increasing) | `registry.gitlab.hpi.de/akita/i/baseline_increasing` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_normal](./baseline_normal) | `registry.gitlab.hpi.de/akita/i/baseline_normal` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_random](./baseline_random) | `registry.gitlab.hpi.de/akita/i/baseline_random` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cblof](./cblof) | `registry.gitlab.hpi.de/akita/i/cblof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cof](./cof) | `registry.gitlab.hpi.de/akita/i/cof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [copod](./copod) | `registry.gitlab.hpi.de/akita/i/copod` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [dae](./dae) (DeNoising Autoencoder) | `registry.gitlab.hpi.de/akita/i/dae` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [damp](./damp) | `registry.gitlab.hpi.de/akita/i/damp` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [dbstream](./dbstream) | `registry.gitlab.hpi.de/akita/i/dbstream` | R 4.0.5 | [`registry.gitlab.hpi.de/akita/i/r4-base`](./0-base-images/r4-base) | unsupervised | multivariate
| [deepant](./deepant) | `registry.gitlab.hpi.de/akita/i/deepant` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [deepnap](./deepnap) | `registry.gitlab.hpi.de/akita/i/deepnap` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [donut](./donut) | `registry.gitlab.hpi.de/akita/i/donut` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [dspot](./dspot) | `registry.gitlab.hpi.de/akita/i/dspot` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [dwt_mlead](./dwt_mlead) | `registry.gitlab.hpi.de/akita/i/dwt_mlead` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [eif](./eif) | `registry.gitlab.hpi.de/akita/i/eif` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [encdec_ad](./encdec_ad) | `registry.gitlab.hpi.de/akita/i/encdec_ad` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [ensemble_gi](./ensemble_gi) | `registry.gitlab.hpi.de/akita/i/ensemble_gi` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [fast_mcd](./fast_mcd) | `registry.gitlab.hpi.de/akita/i/fast_mcd` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [fft](./fft) | `registry.gitlab.hpi.de/akita/i/fft` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [generic_rf](./generic_rf) | `registry.gitlab.hpi.de/akita/i/generic_rf` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [generic_xgb](./generic_xgb) | `registry.gitlab.hpi.de/akita/i/generic_xgb` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [grammarviz3](./grammarviz3) | `registry.gitlab.hpi.de/akita/i/grammarviz3` | Java| [`registry.gitlab.hpi.de/akita/i/java-base`](./0-base-images/java-base) | unsupervised | univariate |
| [grammarviz3_multi](./grammarviz3_multi) | `registry.gitlab.hpi.de/akita/i/grammarviz3_multi` | Java| [`registry.gitlab.hpi.de/akita/i/java-base`](./0-base-images/java-base) | unsupervised | multivariate |
| [hbos](./hbos) | `registry.gitlab.hpi.de/akita/i/hbos` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [health_esn](./health_esn) | `registry.gitlab.hpi.de/akita/i/health_esn` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [hif](./hif) | `registry.gitlab.hpi.de/akita/i/hif` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [hotsax](./hotsax) | `registry.gitlab.hpi.de/akita/i/hotsax` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [hybrid_knn](./hybrid_knn) | `registry.gitlab.hpi.de/akita/i/hybrid_knn` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [if_lof](./if_lof) | `registry.gitlab.hpi.de/akita/i/if_lof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [iforest](./iforest) | `registry.gitlab.hpi.de/akita/i/iforest` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [img_embedding_cae](./img_embedding_cae) | `registry.gitlab.hpi.de/akita/i/img_embedding_cae` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [kmeans](./kmeans) | `registry.gitlab.hpi.de/akita/i/kmeans` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [knn](./knn) | `registry.gitlab.hpi.de/akita/i/knn` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [laser_dbn](./laser_dbn) | `registry.gitlab.hpi.de/akita/i/laser_dbn` | Python 3.7 |[`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [left_stampi](./left_stampi) | `registry.gitlab.hpi.de/akita/i/left_stampi` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [lof](./lof) | `registry.gitlab.hpi.de/akita/i/lof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [lstm_ad](./lstm_ad) | `registry.gitlab.hpi.de/akita/i/lstm_ad` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [lstm_vae](./lstm_vae) | `registry.gitlab.hpi.de/akita/i/lstm_vae` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [median_method](./median_method) | `registry.gitlab.hpi.de/akita/i/median_method` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [mscred](./mscred) | `registry.gitlab.hpi.de/akita/i/mscred` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [mstamp](./mstamp) | `registry.gitlab.hpi.de/akita/i/mstamp` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [mtad_gat](./mtad_gat) | `registry.gitlab.hpi.de/akita/i/mtad_gat` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [multi_hmm](./multi_hmm) | `registry.gitlab.hpi.de/akita/i/multi_hmm` | Python 3.7 |[`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [multi_subsequence_lof](./multi_subsquence_lof) | `registry.gitlab.hpi.de/akita/i/multi_subsequence_lof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [mvalmod](./mvalmod) | `registry.gitlab.hpi.de/akita/i/mvalmod` | R 3.5.2 | [`registry.gitlab.hpi.de/akita/i/tsmp`](./0-base-images/tsmp) -> [`registry.gitlab.hpi.de/akita/i/r-base`](./0-base-images/r-base) | unsupervised | multivariate |
| [norma](./norma) (_restricted access_) | `registry.gitlab.hpi.de/akita/i/norma` | Python 3.7 |[`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [normalizing_flows](./normalizing_flows) | `registry.gitlab.hpi.de/akita/i/normalizing_flows` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [novelty_svr](./novelty_svr) | `registry.gitlab.hpi.de/akita/i/novelty_svr` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [numenta_htm](./numenta_htm) | `registry.gitlab.hpi.de/akita/i/numenta_htm` | Python 2.7 |[`registry.gitlab.hpi.de/akita/i/python2-base`](./0-base-images/python2-base) | unsupervised | univariate |
| [ocean_wnn](./ocean_wnn) | `registry.gitlab.hpi.de/akita/i/ocean_wnn` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [omnianomaly](./omnianomaly) | `registry.gitlab.hpi.de/akita/i/omnianomaly` | Python 3.6 |[`registry.gitlab.hpi.de/akita/i/python36-base`](./0-base-images/python36-base) | semi-supervised | multivariate |
| [pcc](./pcc) | `registry.gitlab.hpi.de/akita/i/pcc` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [pci](./pci) | `registry.gitlab.hpi.de/akita/i/pci` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [phasespace_svm](./phasespace_svm) | `registry.gitlab.hpi.de/akita/i/phasespace_svm` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [pst](./pst) | `registry.gitlab.hpi.de/akita/i/pst` | R 3.5.2 | [`registry.gitlab.hpi.de/akita/i/r-base`](./0-base-images/r-base) | |
| [random_black_forest](./random_black_forest) | `registry.gitlab.hpi.de/akita/i/random_black_forest` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [robust_pca](./robust_pca) | `registry.gitlab.hpi.de/akita/i/robust_pca` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [sand](./sand) (_restricted access_) | `registry.gitlab.hpi.de/akita/i/sand` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sarima](./sarima) | `registry.gitlab.hpi.de/akita/i/sarima` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [series2graph](./series2graph)  (_restricted access_) | `registry.gitlab.hpi.de/akita/i/series2graph` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [s_h_esd](./s_h_esd) | `registry.gitlab.hpi.de/akita/i/s_h_esd` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr](./sr) | `registry.gitlab.hpi.de/akita/i/sr` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr_cnn](./sr_cnn) | `registry.gitlab.hpi.de/akita/i/sr_cnn` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [ssa](./ssa) (_restricted access_) | `registry.gitlab.hpi.de/akita/i/ssa` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [stamp](./stamp) | `registry.gitlab.hpi.de/akita/i/stamp` | R 3.5.2 | [`registry.gitlab.hpi.de/akita/i/tsmp`](./0-base-images/tsmp) -> [`registry.gitlab.hpi.de/akita/i/r-base`](./0-base-images/r-base) | unsupervised | univariate |
| [stomp](./stomp) | `registry.gitlab.hpi.de/akita/i/stomp` | R 3.5.2 | [`registry.gitlab.hpi.de/akita/i/tsmp`](./0-base-images/tsmp) -> [`registry.gitlab.hpi.de/akita/i/r-base`](./0-base-images/r-base) | unsupervised | univariate |
| [subsequence_fast_mcd](./subsequence_fast_mcd) | `registry.gitlab.hpi.de/akita/i/subsequence_fast_mcd` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [subsequence_knn](./subsequence_knn) | `registry.gitlab.hpi.de/akita/i/subsequence_knn` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [subsequence_if](./subsequence_if) | `registry.gitlab.hpi.de/akita/i/subsequence_if` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [subsequence_lof](./subsequence_lof) | `registry.gitlab.hpi.de/akita/i/subsequence_lof` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/pyod`](./0-base-images/pyod) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [tanogan](./tanogan) | `registry.gitlab.hpi.de/akita/i/tanogan` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-torch) -> [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [tarzan](./tarzan) | `registry.gitlab.hpi.de/akita/i/tarzan` | Python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [telemanom](./telemanom) | `registry.gitlab.hpi.de/akita/i/telemanom` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [torsk](./torsk) | `registry.gitlab.hpi.de/akita/i/torsk` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [triple_es](./triple_es) | `registry.gitlab.hpi.de/akita/i/triple_es` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [ts_bitmap](./ts_bitmap) | `registry.gitlab.hpi.de/akita/i/ts_bitmap` | python 3.7 | [`registry.gitlab.hpi.de/akita/i/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [valmod](./valmod) | `registry.gitlab.hpi.de/akita/i/valmod` | R 3.5.2 | [`registry.gitlab.hpi.de/akita/i/tsmp`](./0-base-images/tsmp) -> [`registry.gitlab.hpi.de/akita/i/r-base`](./0-base-images/r-base) | unsupervised | univariate |

## Usage

### Prerequisites

You need the following tools installed on your development machine:

- git
- docker
- access to this repository

Please make yourself familiar with the contents of this repository and read this document carefully!

### Testing an algorithm and its TimeEval integration

Testing an algorithm locally can be done in two different ways:

1. Test the algorithm's code directly (using the tools provided by the programming language)
2. Test the algorithm within its docker container

The first option is specific to the programming language, so we won't cover it here.

Each algorithm in this repository will be bundled in a self-contained Docker image so that it can be executed with a single command and no additional dependencies must be installed.
This allows you to test the algorithm without installing its dependencies on your machine.
The only requirement is a (x86-)Docker runtime.
Follow the below steps to test your algorithm using Docker:

1. **Prepare base image**
   You'll need the required base Docker image to build your algorithm's image.
   If you find yourself situated in the HPI network (either VPN or physically), you are able to pull the docker images from our docker repository `registry.gitlab.hpi.de/akita/i/`.
   If this is not the case you have to build the base images yourself as follows:

   - change to the `0-base-images` folder: `cd 0-base-images`
   - build your desired base image, e.g. `docker build -t registry.gitlab.hpi.de/akita/i/python3-base:0.2.5 ./python3-base`
   - (optionally: build derived base image, e.g. `docker build -t registry.gitlab.hpi.de/akita/i/pyod:0.2.5 ./pyod`)
   - now you can build your algorithm image from the base image (see next item)

2. **Build algorithm image**
   Next, you'll need to build the algorithm's Docker image.
   It is based on the previously built base image and contains the algorithm-specific source code.

   - Change to the root directory of the `timeeval-algorithms`-repository.
   - build the algorithm image, e.g. `docker build -t registry.gitlab.hpi.de/akita/i/lof ./lof`

3. **Train your algorithm (optional)**
   If your algorithm is supervised or semi-supervised, execute the following command to perform the training step:

   ```bash
   mkdir -p 2-results
   docker run --rm \
       -v $(pwd)/1-data:/data:ro \
       -v $(pwd)/2-results:/results:rw \
   #    -e LOCAL_UID=<current user id> \
   #    -e LOCAL_GID=<current groupid> \
     registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest execute-algorithm '{ \
       "executionType": "train", \
       "dataInput": "/data/dataset.csv", \
       "dataOutput": "/results/anomaly_scores.ts", \
       "modelInput": "/results/model.pkl", \
       "modelOutput": "/results/model.pkl", \
       "customParameters": {} \
     }'
   ```

   Be warned that the result and model files will be written to the `2-results`-directory as the root-user if you do no pass the optional environment variables `LOCAL_UID` and `LOCAL_GID` to the container.

4. **Execute your algorithm**
   Run the following command to perform the execution step of your algorithm:

   ```bash
   mkdir -p 2-results
   docker run --rm \
       -v $(pwd)/1-data:/data:ro \
       -v $(pwd)/2-results:/results:rw \
   #    -e LOCAL_UID=<current user id> \
   #    -e LOCAL_GID=<current groupid> \
     registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest execute-algorithm '{ \
       "executionType": "execute", \
       "dataInput": "/data/dataset.csv", \
       "dataOutput": "/results/anomaly_scores.ts", \
       "modelInput": "/results/model.pkl", \
       "modelOutput": "/results/model.pkl", \
       "customParameters": {} \
     }'
   ```

   Be warned that the result and model files will be written to the `2-results`-directory as the root-user if you do no pass the optional environment variables `LOCAL_UID` and `LOCAL_GID` to the container.

## Additional information

### TimeEval base Docker images

To benefit from Docker layer caching and to reduce code duplication (DRY!), we decided to put common functionality in so-called base images.
The following is taken care of by base images:

- Provide system (OS and common OS tools)
- Provide language runtime (e.g. python3, java8)
- Provide common libraries / algorithm dependencies
- Define volumes for IO
- Define Docker entrypoint script (performs initial container setup before the algorithm is executed)

Please consider the folder [0-base-images](./0-base-images) for all available base images.

### TimeEval algorithm interface

TimeEval uses a common interface to execute all the algorithms in this repository.
This means that the algorithms' input, output, and parametrization is equal for all TimeEval algorithms.

#### Execution and parametrization

All algorithms are executed by creating a Docker container using their Docker image and running it.
The base images take care of the container startup and they call the main algorithm file with a single positional parameter.
This parameter contains a String-representation of the algorithm configuration JSON.
Example parameter JSON (2021-02-03):

```python
{
  "executionType": 'train' | 'execute',
  "dataInput": string,   # example: "path/to/dataset.csv",
  "dataOutput": string,  # example: "path/to/results.csv",
  "modelInput": string,  # example: "/path/to/model.pkl",
  "modelOutput": string, # example: "/path/to/model.pkl",
  "customParameters": dict
}
```

#### Custom algorithm parameters

All algorithm hyper parameters described in the paper are exposed via the `customParameters` configuration option.
This allows us to set those parameters from TimeEval.

> **Attention!**
>
> TimeEval does **not** parse a `manifest.json` file to get the custom parameters' types and default values.
> We expect the users of TimeEval to be familiar with the algorithms, so that they can specify the required parameters manually.
> However, we require each algorithm to be executable without specifying any custom parameters (especially for testing purposes).
> Therefore, **please provide sensible default parameters for all custom parameters within the method's code**.
>
> Please add a `manifest.json` file to your algorithm anyway to aid the integration into UltraMine and for user information.
>
> If your algorithm does not use the default parameters automatically and expects them to be provided, your algorithm will fail during runtime if no parameters are provided by the TimeEval user.

#### Input and output

Input and output for an algorithm is handled via bind-mounting files and folders into the Docker container.

All **input data**, such as the training dataset and the test dataset, are mounted read-only to the `/data`-folder of the container.
The configuration options `dataInput` and `modelInput` reflect this with the correct path to the dataset (e.g. `{ "dataInput": "/data/dataset.test.csv" }`).

All **output** of your algorithm should be written to the `/results`-folder.
This is also reflected in the configuration options with the correct paths for `dataOutput` and `modelOutput` (e.g. `{ "dataOutput": "/results/anomaly_scores.csv" }`).
The `/results`-folder is also bind-mounted to the algorithm container - but writable -, so that TimeEval can access the results after your algorithm finished.
An algorithm can also use this folder to write persistent log and debug information.

**Temporary files** and data of an algorithm are written to the current working directory (currently this is `/app`) or the temporary directory `/tmp` within the Docker container.

#### Example calls

The following Docker command represents the way how TimeEval executes your algorithm image:

```bash
docker run --rm \
    -v $(pwd)/1-data:/data:ro \
    -v $(pwd)/2-results:/results:rw \
#    -e LOCAL_UID=<current user id> \
#    -e LOCAL_GID=<groupid of akita group> \
#    <resource restrictions> \
  registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest execute-algorithm '{ \
    "executionType": "train", \
    "dataInput": "/data/dataset.csv", \
    "modelInput": "/data/model.pkl", \
    "dataOutput": "/results/anomaly_scores.ts", \
    "modelOutput": "/results/model.pkl", \
    "customParameters": {} \
  }'
```

This is translated to the following call within the container:

```bash
mkdir results
docker run --rm \
    -v $(pwd)/1-data:/data:ro \
    -v $(pwd)/2-results:/results:rw \
  registry.gitlab.hpi.de/akita/i/<your_algorithm>:latest bash
# now, within the container
<python | java -jar | Rscript> $ALGORITHM_MAIN '{ \
  "executionType": "train", \
  "dataInput": "/data/dataset.csv", \
  "modelInput": "/data/model.pkl", \
  "dataOutput": "/results/anomaly_scores.ts", \
  "modelOutput": "/results/model.pkl", \
  "customParameters": {} \
}'
```
