# Docker Images

AKITA Docker Images used as base for algorithms.
Images are automatically build and published to the internal Docker registry at `mut:5000`.

## Overview

Base images:

| Name/Folder | Image | Usage |
| :--- | :---- | :---- |
| python2-base | `registry.gitlab.hpi.de/akita/i/python2-base` | Base image for TimeEval methods that use python2 (version 2.7); includes default python packages, see [`requirements.txt`](./python2-base/requirements.txt). |
| python3-base | `registry.gitlab.hpi.de/akita/i/python3-base` | Base image for TimeEval methods that use python3 (version 3.7.9); includes default python packages, see [`requirements.txt`](./python3-base/requirements.txt). |
| python36-base | `registry.gitlab.hpi.de/akita/i/python36-base` | Base image for TimeEval methods that use python3.6 (version 3.6.13); includes default python packages, see [`requirements.txt`](./python36-base/requirements.txt). |
| r-base | `registry.gitlab.hpi.de/akita/i/r-base` | Base image for TimeEval methods that use R (version 3.5.2-1). |
| r4-base | `registry.gitlab.hpi.de/akita/i/r4-base` | Base image for TimeEval methods that use R (version 4.0.5). |
| java-base | `registry.gitlab.hpi.de/akita/i/java-base` | Base image for TimeEval methods that use Java (JRE 11.0.10). |

Derived base images:

| Name/Folder | Image | Usage |
| :--- | :---- | :---- |
| tsmp | `registry.gitlab.hpi.de/akita/i/tsmp` | Base image for TimeEval methods that use the matrix profile R package [`tsmp`](https://github.com/matrix-profile-foundation/tsmp); is based on `registry.gitlab.hpi.de/akita/i/r-base`. |
| pyod | `registry.gitlab.hpi.de/akita/i/pyod` | Base image for TimeEval methods that are based on the [`pyod`](https://github.com/yzhao062/pyod) library; is based on `registry.gitlab.hpi.de/akita/i/python3-base` |
| timeeval-test-algorithm | `registry.gitlab.hpi.de/akita/i/timeeval-test-algorithm` | Test image for TimeEval tests that use docker; is based on `registry.gitlab.hpi.de/akita/i/python3-base`. |
| python3-torch | `registry.gitlab.hpi.de/akita/i/python3-torch` | Base image for TimeEval methods that use python3 (version 3.7.9) and PyTorch (version 1.7.1); includes default python packages, see [`requirements.txt`](./python3-base/requirements.txt), and torch; is based on `registry.gitlab.hpi.de/akita/i/python3-base`. |

## Accessing the AKITA Docker registry

The AKITA Docker registry at `registry.gitlab.hpi.de/akita/i` is only accessible for HPI staff and students.

As an HPI member, please contact the maintainers of this repository for access credentials!
