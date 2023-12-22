<div align="center">
<img width="130px" src="./timeeval-algorithms.png" alt="TimeEval logo"/>
<h1 align="center">TimeEval Algorithms</h1>
<p>
  Time Series Anomaly Detection Algorithms for TimeEval.
</p>
</div>

## Description

This repository contains a collection of containerized (dockerized) time series anomaly detection methods that can easily be evaluated using [TimeEval](https://github.com/TimeEval/TimeEval).
Some of the algorithm's source code is access restricted and we just provide the TimeEval stubs and manifests.
We are happy to share our TimeEval adaptations of excluded algorithms upon request, if the original authors approve this.

Each folder contains the implementation of an algorithm that is built into a runnable Docker container using GitHub Actions.
We host the algorithm Docker images on GitHub.
Thus, the namespace prefix (repository) for the Docker images is `ghcr.io/timeeval/`.

## Overview

| Algorithm (folder) | Image | Language | Base image | Learning Type | Input Dimensionality |
| :----------------- | :---- | :------- | :--------- | :------------ | :------------------- |
| [arima](./arima) (_restricted access_) | [`ghcr.io/timeeval/arima`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/arima) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [autoencoder](./autoencoder) | [`ghcr.io/timeeval/autoencoder`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/autoencoder) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [bagel](./bagel) | [`ghcr.io/timeeval/bagel`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/bagel) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [baseline_increasing](./baseline_increasing) | [`ghcr.io/timeeval/baseline_increasing`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/baseline_increasing) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_normal](./baseline_normal) | [`ghcr.io/timeeval/baseline_normal`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/baseline_normal) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_random](./baseline_random) | [`ghcr.io/timeeval/baseline_random`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/baseline_random) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cblof](./cblof) | [`ghcr.io/timeeval/cblof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/cblof) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cof](./cof) | [`ghcr.io/timeeval/cof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/cof) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [copod](./copod) | [`ghcr.io/timeeval/copod`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/copod) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [dae](./dae) (DeNoising Autoencoder) | [`ghcr.io/timeeval/dae`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/dae) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [damp](./damp) | [`ghcr.io/timeeval/damp`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/damp) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [dbstream](./dbstream) | [`ghcr.io/timeeval/dbstream`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/dbstream) | R 4.2.0 | [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | unsupervised | multivariate
| [deepant](./deepant) | [`ghcr.io/timeeval/deepant`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/deepant) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [deepnap](./deepnap) | [`ghcr.io/timeeval/deepnap`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/deepnap) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [donut](./donut) | [`ghcr.io/timeeval/donut`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/donut) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [dspot](./dspot) | [`ghcr.io/timeeval/dspot`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/dspot) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [dwt_mlead](./dwt_mlead) | [`ghcr.io/timeeval/dwt_mlead`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/dwt_mlead) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [eif](./eif) | [`ghcr.io/timeeval/eif`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/eif) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [encdec_ad](./encdec_ad) | [`ghcr.io/timeeval/encdec_ad`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/encdec_ad) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [ensemble_gi](./ensemble_gi) | [`ghcr.io/timeeval/ensemble_gi`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/ensemble_gi) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [fast_mcd](./fast_mcd) | [`ghcr.io/timeeval/fast_mcd`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/fast_mcd) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [fft](./fft) | [`ghcr.io/timeeval/fft`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/fft) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [generic_rf](./generic_rf) | [`ghcr.io/timeeval/generic_rf`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/generic_rf) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [generic_xgb](./generic_xgb) | [`ghcr.io/timeeval/generic_xgb`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/generic_xgb) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [grammarviz3](./grammarviz3) | [`ghcr.io/timeeval/grammarviz3`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/grammarviz3) | Java| [`ghcr.io/timeeval/java-base`](./0-base-images/java-base) | unsupervised | univariate |
| [grammarviz3_multi](./grammarviz3_multi) | [`ghcr.io/timeeval/grammarviz3_multi`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/grammarviz3_multi) | Java| [`ghcr.io/timeeval/java-base`](./0-base-images/java-base) | unsupervised | multivariate |
| [hbos](./hbos) | [`ghcr.io/timeeval/hbos`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/hbos) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [health_esn](./health_esn) | [`ghcr.io/timeeval/health_esn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/health_esn) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [hif](./hif) | [`ghcr.io/timeeval/hif`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/hif) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [hotsax](./hotsax) | [`ghcr.io/timeeval/hotsax`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/hotsax) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [hybrid_knn](./hybrid_knn) | [`ghcr.io/timeeval/hybrid_knn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/hybrid_knn) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [if_lof](./if_lof) | [`ghcr.io/timeeval/if_lof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/if_lof) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [iforest](./iforest) | [`ghcr.io/timeeval/iforest`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/iforest) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [img_embedding_cae](./img_embedding_cae) | [`ghcr.io/timeeval/img_embedding_cae`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/img_embedding_cae) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [kmeans](./kmeans) | [`ghcr.io/timeeval/kmeans`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/kmeans) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [knn](./knn) | [`ghcr.io/timeeval/knn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/knn) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [laser_dbn](./laser_dbn) | [`ghcr.io/timeeval/laser_dbn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/laser_dbn) | Python 3.7 |[`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [left_stampi](./left_stampi) | [`ghcr.io/timeeval/left_stampi`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/left_stampi) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [lof](./lof) | [`ghcr.io/timeeval/lof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/lof) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [lstm_ad](./lstm_ad) | [`ghcr.io/timeeval/lstm_ad`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/lstm_ad) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [lstm_vae](./lstm_vae) | [`ghcr.io/timeeval/lstm_vae`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/lstm_vae) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [median_method](./median_method) | [`ghcr.io/timeeval/median_method`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/median_method) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [mscred](./mscred) | [`ghcr.io/timeeval/mscred`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/mscred) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [mstamp](./mstamp) | [`ghcr.io/timeeval/mstamp`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/mstamp) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [mtad_gat](./mtad_gat) | [`ghcr.io/timeeval/mtad_gat`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/mtad_gat) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [multi_hmm](./multi_hmm) | [`ghcr.io/timeeval/multi_hmm`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/multi_hmm) | Python 3.7 |[`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [multi_norma](./multi_norma) (_restricted access_) | [`ghcr.io/timeeval/multi_norma`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/multi_norma) | Python 3.7 |[`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [multi_subsequence_lof](./multi_subsquence_lof) | [`ghcr.io/timeeval/multi_subsequence_lof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/multi_subsequence_lof) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [mvalmod](./mvalmod) | [`ghcr.io/timeeval/mvalmod`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/mvalmod) | R 4.2.0 | [`ghcr.io/timeeval/tsmp`](./1-intermediate-images/tsmp) -> [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | unsupervised | multivariate |
| [norma](./norma) (_restricted access_) | [`ghcr.io/timeeval/norma`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/norma) | Python 3.7 |[`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [normalizing_flows](./normalizing_flows) | [`ghcr.io/timeeval/normalizing_flows`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/normalizing_flows) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [novelty_svr](./novelty_svr) | [`ghcr.io/timeeval/novelty_svr`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/novelty_svr) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [numenta_htm](./numenta_htm) | [`ghcr.io/timeeval/numenta_htm`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/numenta_htm) | Python 2.7 |[`ghcr.io/timeeval/python2-base`](./0-base-images/python2-base) | unsupervised | univariate |
| [ocean_wnn](./ocean_wnn) | [`ghcr.io/timeeval/ocean_wnn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/ocean_wnn) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [omnianomaly](./omnianomaly) | [`ghcr.io/timeeval/omnianomaly`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/omnianomaly) | Python 3.6 |[`ghcr.io/timeeval/python36-base`](./0-base-images/python36-base) | semi-supervised | multivariate |
| [pcc](./pcc) | [`ghcr.io/timeeval/pcc`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/pcc) | Python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [pci](./pci) | [`ghcr.io/timeeval/pci`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/pci) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [phasespace_svm](./phasespace_svm) | [`ghcr.io/timeeval/phasespace_svm`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/phasespace_svm) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [pst](./pst) | [`ghcr.io/timeeval/pst`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/pst) | R 4.2.0 | [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | |
| [random_black_forest](./random_black_forest) | [`ghcr.io/timeeval/random_black_forest`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/random_black_forest) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [robust_pca](./robust_pca) | [`ghcr.io/timeeval/robust_pca`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/robust_pca) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [sand](./sand) (_restricted access_) | [`ghcr.io/timeeval/sand`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/sand) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sarima](./sarima) | [`ghcr.io/timeeval/sarima`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/sarima) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [series2graph](./series2graph)  (_restricted access_) | [`ghcr.io/timeeval/series2graph`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/series2graph) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [s_h_esd](./s_h_esd) | [`ghcr.io/timeeval/s_h_esd`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/s_h_esd) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr](./sr) | [`ghcr.io/timeeval/sr`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/sr) | Python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr_cnn](./sr_cnn) | [`ghcr.io/timeeval/sr_cnn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/sr_cnn) | Python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [ssa](./ssa) (_restricted access_) | [`ghcr.io/timeeval/ssa`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/ssa) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [stamp](./stamp) | [`ghcr.io/timeeval/stamp`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/stamp) | R 4.2.0 | [`ghcr.io/timeeval/tsmp`](./1-intermediate-images/tsmp) -> [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | unsupervised | univariate |
| [stomp](./stomp) | [`ghcr.io/timeeval/stomp`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/stomp) | R 4.2.0 | [`ghcr.io/timeeval/tsmp`](./1-intermediate-images/tsmp) -> [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | unsupervised | univariate |
| [subsequence_fast_mcd](./subsequence_fast_mcd) | [`ghcr.io/timeeval/subsequence_fast_mcd`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/subsequence_fast_mcd) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [subsequence_knn](./subsequence_knn) | [`ghcr.io/timeeval/subsequence_knn`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/subsequence_knn) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [subsequence_if](./subsequence_if) | [`ghcr.io/timeeval/subsequence_if`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/subsequence_if) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [subsequence_lof](./subsequence_lof) | [`ghcr.io/timeeval/subsequence_lof`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/subsequence_lof) | python 3.7 | [`ghcr.io/timeeval/pyod`](./1-intermediate-images/pyod) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [tanogan](./tanogan) | [`ghcr.io/timeeval/tanogan`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/tanogan) | python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-torch) -> [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [tarzan](./tarzan) | [`ghcr.io/timeeval/tarzan`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/tarzan) | Python 3.7 | [`ghcr.io/timeeval/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [telemanom](./telemanom) | [`ghcr.io/timeeval/telemanom`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/telemanom) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [torsk](./torsk) | [`ghcr.io/timeeval/torsk`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/torsk) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [triple_es](./triple_es) | [`ghcr.io/timeeval/triple_es`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/triple_es) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [ts_bitmap](./ts_bitmap) | [`ghcr.io/timeeval/ts_bitmap`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/ts_bitmap) | python 3.7 | [`ghcr.io/timeeval/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [valmod](./valmod) | [`ghcr.io/timeeval/valmod`](https://github.com/TimeEval/TimeEval-algorithms/pkgs/container/valmod) | R 4.2.0 | [`ghcr.io/timeeval/tsmp`](./1-intermediate-images/tsmp) -> [`ghcr.io/timeeval/r4-base`](./0-base-images/r4-base) | unsupervised | univariate |

## Usage

### Use the published algorithm images

Please always use a version-tagged Docker image for your algorithms to ensure reproducibility!

You can pull the TimeEval algorithm images directly from the GitHub registry.
This registry does not require authentication.

```bash
docker pull ghcr.io/timeeval/<algorithm_name>:0.3.0
```

### Build the algorithm images

Each algorithm in this repository is bundled in a self-contained Docker image so that it can be executed with a single command and no additional dependencies must be installed.
This allows you to test the algorithm without installing its dependencies on your machine.
The only requirement is a (x86-)Docker runtime.

In the following, we assume that you want to build the Docker image for the [`lof`](./lof)-algorithm.

> :warning: **Use correct version tags!**
>
> Please tag the base and intermediate images with the correct version.
> You can find the required version for each algorithm image in its `Dockerfile`.
> E.g. for [`lof`](./lof/Dockerfile) the version for `pyod` must be `0.3.0` (as of 2023-12-16):
>
> ```Dockerfile
> FROM ghcr.io/timeeval/pyod:0.3.0
> ```

#### Prerequisites

You need the following tools installed on your development machine:

- git
- docker
- access to this repository
- (optionally) Docker BuildKit

Please make yourself familiar with the [concepts of TimeEval](https://timeeval.readthedocs.io/en/latest/concepts), and read the [TimeEval documentation](https://timeeval.readthedocs.io) and this document carefully!

#### 1. Prepare base image

You'll need the required base Docker images to build your algorithm's image.
You can either pull the base image from the registry or build it yourself.
In this guide, we assume that you want to build all Docker images locally.

1. Clone this repository and change to its root folder

   ```bash
   git clone https://github.com/TimeEval/TimeEval-algorithms.git
   cd TimeEval-algorithms
   ```

2. Change to the `0-base-images` folder:

   ```bash
   cd 0-base-images
   ```

3. Build your desired base image:

   ```bash
   docker build -t ghcr.io/timeeval/python3-base:0.3.0 python3-base
   ```

#### Prepare intermediate image (optional)

Because the algorithm `lof` depends on an intermediate image, we, first, need to build the required intermediate image `pyod`.
Please see the table in this repository's README for the dependencies.

You can build the intermediate image `pyod` using these commands:

```bash
cd ../1-intermediate-images
docker build -t ghcr.io/timeeval/pyod:0.2.5 pyod
```

#### Build algorithm image

Once you have built all dependent images, you can build your algorithm image from the base image(s):

1. Change to the repository's root directory:

   ```bash
   cd ..
   ```

2. Build the Docker image for your algorithm (`lof` in this case):

   ```bash
   docker build -t ghcr.io/timeeval/lof:0.3.0 ./lof
   ```

### Testing an algorithm and its TimeEval integration

Testing an algorithm locally can be done in two different ways:

1. Test the algorithm's code directly (using the tools provided by the programming language)
2. Test the algorithm within its docker container

The first option is specific to the programming language, so we won't cover it here.

Each algorithm in this repository will be bundled in a self-contained Docker image so that it can be executed with a single command and no additional dependencies must be installed.
This allows you to test the algorithm without installing its dependencies on your machine.
The only requirement is a (x86-)Docker runtime.
Follow the below steps to test your algorithm using Docker (examples assume that you want to build the image for the LOF algorithm):

1. **Pull or build the algorithm image**
   We refer the reader to the previous section for detailed instructions.

2. **Train your algorithm (optional)**
   If your algorithm is supervised or semi-supervised, execute the following command to perform the training step (_not necessary for LOF_):

   ```bash
   mkdir -p results
   docker run --rm \
       -v $(pwd)/data:/data:ro \
       -v $(pwd)/results:/results:rw \
   #    -e LOCAL_UID=<current user id> \
   #    -e LOCAL_GID=<current groupid> \
     ghcr.io/timeeval/<your_algorithm>:latest execute-algorithm '{
       "executionType": "train",
       "dataInput": "/data/dataset.csv",
       "dataOutput": "/results/anomaly_scores.ts",
       "modelInput": "/results/model.pkl",
       "modelOutput": "/results/model.pkl",
       "customParameters": {}
     }'
   ```

   Be warned that the result and model files will be written to the `results`-directory as the root-user if you do not pass the optional environment variables `LOCAL_UID` and `LOCAL_GID` to the container.

3. **Execute your algorithm**
   Run the following command to perform the execution step of your algorithm:

   ```bash
   mkdir -p results
   TIMEEVAL_ALGORITHM=lof
   docker run --rm \
       -v $(pwd)/data:/data:ro \
       -v $(pwd)/results:/results:rw \
   #    -e LOCAL_UID=<current user id> \
   #    -e LOCAL_GID=<current groupid> \
     ghcr.io/timeeval/${TIMEEVAL_ALGORITHM}:latest execute-algorithm '{
       "executionType": "execute",
       "dataInput": "/data/dataset.csv",
       "dataOutput": "/results/anomaly_scores.ts",
       "modelInput": "/results/model.pkl",
       "modelOutput": "/results/model.pkl",
       "customParameters": {}
     }'
   ```

   Be warned that the result and model files will be written to the `results`-directory as the root-user if you do not pass the optional environment variables `LOCAL_UID` and `LOCAL_GID` to the container.
