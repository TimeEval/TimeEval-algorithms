# TimeEval Algorithms

[![pipeline status](https://gitlab.hpi.de/akita/timeeval-algorithms/badges/main/pipeline.svg)](https://gitlab.hpi.de/akita/timeeval-algorithms/-/commits/main)

This repository contains a collection of containerized (dockerized) time series anomaly detection methods that can easily be evaluated using [TimeEval](https://github.com/HPI-Information-Systems/TimeEval).

Each folder contains the implementation of an algorithm that will be build into a runnable Docker container using CI.
The namespace prefix (repository) for the built Docker images is `mut:5000/akita/`.

## Overview

| Algorithm (folder) | Image | Language | Base image | Learning Type | Input Dimensionality |
| :----------------- | :---- | :------- | :--------- | :------------ | :------------------- |
| [arima](./arima) | `mut:5000/akita/arima` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [autoencoder](./autoencoder) | `mut:5000/akita/autoencoder` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [bagel](./bagel) | `mut:5000/akita/bagel` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [baseline_increasing](./baseline_increasing) | `mut:5000/akita/baseline_increasing` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_normal](./baseline_normal) | `mut:5000/akita/baseline_normal` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [baseline_random](./baseline_random) | `mut:5000/akita/baseline_random` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cblof](./cblof) | `mut:5000/akita/cblof` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [cof](./cof) | `mut:5000/akita/cof` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [copod](./copod) | `mut:5000/akita/copod` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [dae](./dae) (DeNoising Autoencoder) | `mut:5000/akita/dae` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [dbstream](./dbstream) | `mut:5000/akita/dbstream` | R 4.0.5 | [`mut:5000/akita/r4-base`](./0-base-images/r4-base) | unsupervised | multivariate
| [deepant](./deepant) | `mut:5000/akita/deepant` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [deepnap](./deepnap) | `mut:5000/akita/deepnap` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [donut](./donut) | `mut:5000/akita/donut` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [dspot](./dspot) | `mut:5000/akita/dspot` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [dwt_mlead](./dwt_mlead) | `mut:5000/akita/dwt_mlead` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [eif](./eif) | `mut:5000/akita/eif` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [encdec_ad](./encdec_ad) | `mut:5000/akita/encdec_ad` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [ensemble_gi](./ensemble_gi) | `mut:5000/akita/ensemble_gi` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [fast_mcd](./fast_mcd) | `mut:5000/akita/fast_mcd` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [fft](./fft) | `mut:5000/akita/fft` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [generic_rf](./generic_rf) | `mut:5000/akita/generic_rf` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [generic_xgb](./generic_xgb) | `mut:5000/akita/generic_xgb` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [grammarviz3](./grammarviz3) | `mut:5000/akita/grammarviz3` | Java| [`mut:5000/akita/java-base`](./0-base-images/java-base) | unsupervised | univariate |
| [hbos](./hbos) | `mut:5000/akita/hbos` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [health_esn](./health_esn) | `mut:5000/akita/health_esn` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [hif](./hif) | `mut:5000/akita/hif` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [hotsax](./hotsax) | `mut:5000/akita/hotsax` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [hybrid_knn](./hybrid_knn) | `mut:5000/akita/hybrid_knn` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [if_lof](./if_lof) | `mut:5000/akita/if_lof` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [iforest](./iforest) | `mut:5000/akita/iforest` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [img_embedding_cae](./img_embedding_cae) | `mut:5000/akita/img_embedding_cae` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [kmeans](./kmeans) | `mut:5000/akita/kmeans` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [knn](./knn) | `mut:5000/akita/knn` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [laser_dbn](./laser_dbn) | `mut:5000/akita/laser_dbn` | Python 3.7 |[`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [left_stampi](./left_stampi) | `mut:5000/akita/left_stampi` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [lof](./lof) | `mut:5000/akita/lof` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [lstm_ad](./lstm_ad) | `mut:5000/akita/lstm_ad` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [lstm_vae](./lstm_vae) | `mut:5000/akita/lstm_vae` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [median_method](./median_method) | `mut:5000/akita/median_method` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [mscred](./mscred) | `mut:5000/akita/mscred` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [mtad_gat](./mtad_gat) | `mut:5000/akita/mtad_gat` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [multi_hmm](./multi_hmm) | `mut:5000/akita/multi_hmm` | Python 3.7 |[`mut:5000/akita/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [norma](./norma) | `mut:5000/akita/norma` | Python 3.7 |[`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [normalizing_flows](./normalizing_flows) | `mut:5000/akita/normalizing_flows` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | supervised | multivariate |
| [novelty_svr](./novelty_svr) | `mut:5000/akita/novelty_svr` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [numenta_htm](./numenta_htm) | `mut:5000/akita/numenta_htm` | Python 2.7 |[`mut:5000/akita/python2-base`](./0-base-images/python2-base) | unsupervised | univariate |
| [ocean_wnn](./ocean_wnn) | `mut:5000/akita/ocean_wnn` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [omnianomaly](./omnianomaly) | `mut:5000/akita/omnianomaly` | Python 3.6 |[`mut:5000/akita/python36-base`](./0-base-images/python36-base) | semi-supervised | multivariate |
| [pcc](./pcc) | `mut:5000/akita/pcc` | Python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [pci](./pci) | `mut:5000/akita/pci` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [phasespace_svm](./phasespace_svm) | `mut:5000/akita/phasespace_svm` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [pst](./pst) | `mut:5000/akita/pst` | R 3.5.2 | [`mut:5000/akita/r-base`](./0-base-images/r-base) | |
| [random_black_forest](./random_black_forest) | `mut:5000/akita/random_black_forest` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [robust_pca](./robust_pca) | `mut:5000/akita/robust_pca` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [sand](./sand) | `mut:5000/akita/sand` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sarima](./sarima) | `mut:5000/akita/sarima` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [series2graph](./series2graph) | `mut:5000/akita/series2graph` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [s_h_esd](./s_h_esd) | `mut:5000/akita/s_h_esd` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr](./sr) | `mut:5000/akita/sr` | Python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [sr_cnn](./sr_cnn) | `mut:5000/akita/sr_cnn` | Python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [ssa](./ssa) | `mut:5000/akita/ssa` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [stamp](./stamp) | `mut:5000/akita/stamp` | R 3.5.2 | [`mut:5000/akita/tsmp`](./0-base-images/tsmp) -> [`mut:5000/akita/r-base`](./0-base-images/r-base) | unsupervised | univariate |
| [stomp](./stomp) | `mut:5000/akita/stomp` | R 3.5.2 | [`mut:5000/akita/tsmp`](./0-base-images/tsmp) -> [`mut:5000/akita/r-base`](./0-base-images/r-base) | unsupervised | univariate |
| [subsequence_fast_mcd](./subsequence_fast_mcd) | `mut:5000/akita/subsequence_fast_mcd` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | univariate |
| [subsequence_if](./subsequence_if) | `mut:5000/akita/subsequence_if` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [subsequence_lof](./subsequence_lof) | `mut:5000/akita/subsequence_lof` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [subsequence_lof_multi_sum](./subsequence_lof_multi_sum) | `mut:5000/akita/subsequence_lof_multi_sum` | python 3.7 | [`mut:5000/akita/pyod`](./0-base-images/pyod) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [tanogan](./tanogan) | `mut:5000/akita/tanogan` | python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-torch) -> [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [tarzan](./tarzan) | `mut:5000/akita/tarzan` | Python 3.7 | [`mut:5000/akita/python3-torch`](./0-base-images/python3-base) | semi-supervised | univariate |
| [telemanom](./telemanom) | `mut:5000/akita/telemanom` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | semi-supervised | multivariate |
| [torsk](./torsk) | `mut:5000/akita/torsk` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | multivariate |
| [triple_es](./triple_es) | `mut:5000/akita/triple_es` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [ts_bitmap](./ts_bitmap) | `mut:5000/akita/ts_bitmap` | python 3.7 | [`mut:5000/akita/python3-base`](./0-base-images/python3-base) | unsupervised | univariate |
| [valmod](./valmod) | `mut:5000/akita/valmod` | R 3.5.2 | [`mut:5000/akita/tsmp`](./0-base-images/tsmp) -> [`mut:5000/akita/r-base`](./0-base-images/r-base) | unsupervised | univariate |

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
   If you find yourself situated in the HPI network (either VPN or physically), you are able to pull the docker images from our docker repository `mut:5000/akita/`.
   If this is not the case you have to build the base images yourself as follows:

   - change to the `0-base-images` folder: `cd 0-base-images`
   - build your desired base image, e.g. `docker build -t mut:5000/akita/python3-base ./python3-base`
   - (optionally: build derived base image, e.g. `docker build -t mut:5000/akita/pyod ./pyod`)
   - now you can build your algorithm image from the base image (see next item)

2. **Build algorithm image**
   Next, you'll need to build the algorithm's Docker image.
   It is based on the previously built base image and contains the algorithm-specific source code.

   - Change to the root directory of the `timeeval-algorithms`-repository.
   - build the algorithm image, e.g. `docker build -t mut:5000/akita/lof ./lof`

3. **Train your algorithm (optional)**
   If your algorithm is supervised or semi-supervised, execute the following command to perform the training step:

   ```bash
   mkdir -p 2-results
   docker run --rm \
       -v $(pwd)/1-data:/data:ro \
       -v $(pwd)/2-results:/results:rw \
   #    -e LOCAL_UID=<current user id> \
   #    -e LOCAL_GID=<current groupid> \
     mut:5000/akita/<your_algorithm>:latest execute-algorithm '{ \
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
     mut:5000/akita/<your_algorithm>:latest execute-algorithm '{ \
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
  mut:5000/akita/<your_algorithm>:latest execute-algorithm '{ \
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
  mut:5000/akita/<your_algorithm>:latest bash
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
