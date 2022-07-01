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

The AKITA Docker registry at `mut:5000` is only accessible from the HPI internal network.
From outside network use the HPI VPN!

### Web access (browsing only)

We run a small web server that allows browsing the Docker image registry at [http://mut:8080/](http://mut:8080/).
You can use it to explore the Docker images and tags that are hosted at the registry.

> The registry browser can be accessed via HPI internal network, VPN, **and** SOCKS5 proxy connections.

### Docker access

There are two ways to force your Docker client to talk to the AKITA Docker registry:

- Add the registry as an untrusted registry (no TLS)
- Add our self-signed certificate (with TLS)

#### As untrusted registry

You can configure your Docker daemon to talk to untrusted registries by adding the following snippet to your Docker [daemon.json](https://docs.docker.com/config/daemon/#configure-the-docker-daemon) configuration file:

```json
{
  "insecure-registries" : ["mut:5000"]
}
```

> Further info: [Docker docs](https://docs.docker.com/registry/insecure/#deploy-a-plain-http-registry)

#### By adding self-signed certificate

The AKITA Docker registry is listening on HTTPS using a self-signed certificate.
You can add the public certificate to your Docker daemon to verify the authenticity of the registry and to prevent Docker client errors.
This is the public certificate for the `mut:5000` AKITA registry:

```crt
-----BEGIN CERTIFICATE-----
MIIGPzCCBCegAwIBAgIUL0Z7rlCzFjUj0WE3UBqXGakR4WEwDQYJKoZIhvcNAQEL
BQAwgbIxCzAJBgNVBAYTAkRFMRQwEgYDVQQIDAtCcmFuZGVuYnVyZzEQMA4GA1UE
BwwHUG90c2RhbTEgMB4GA1UECgwXSGFzc28gUGxhdHRuZXIgSW5zdGl0dXQxIjAg
BgNVBAsMGUluZm9ybWF0aW9uIFN5c3RlbXMgQ2hhaXIxDDAKBgNVBAMMA211dDEn
MCUGCSqGSIb3DQEJARYYc2ViYXN0aWFuLnNjaG1pZGxAaHBpLmRlMB4XDTIxMDgx
MDExMTQyNloXDTI2MDgxMTExMTQyNlowgbIxCzAJBgNVBAYTAkRFMRQwEgYDVQQI
DAtCcmFuZGVuYnVyZzEQMA4GA1UEBwwHUG90c2RhbTEgMB4GA1UECgwXSGFzc28g
UGxhdHRuZXIgSW5zdGl0dXQxIjAgBgNVBAsMGUluZm9ybWF0aW9uIFN5c3RlbXMg
Q2hhaXIxDDAKBgNVBAMMA211dDEnMCUGCSqGSIb3DQEJARYYc2ViYXN0aWFuLnNj
aG1pZGxAaHBpLmRlMIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAsXNA
jhlxhhFBK+qfZTMsG39CyQgEXjZx2ORQWZNcFd1rQJkNamk1qYXG7XkRBUDiFBeo
nbXt0cD9jC5oOt4U9ROglgo3ORVWqQyd4yVKUKm9rFTrU4u9y2Go3hIyFwKTVHTx
sgl8xh97Q5vml+OPObYiOXlG0CWEVzPDAqZ1V3NBgMxuw85jtX+lEb8FDdYFng0u
XNqSNv4h90UStid5Hcr0CcB1cKn4lSp9QDa8IuBplJlByQqi7z0wt3ih79+sgCWR
3mDhRvp+TS/YKf6x+JWhh/WeRyAdoRIh53LbNYYNIMwL/UQf/JDFk0K94cZuWZyo
rI8hbwuKetoOFv6sXZ/xilSGftd3tQ3fGRkTce0lOyUiv4CuURKUDnCiG+G/GnK5
sXrbTzu/yv8lEEg/nYNzOyDVJljWcZK2Ej4ffAca5RB4TeOiz7h61fFjdtm4HZ/5
EtWa4YNXH5a7XnFSaQDRF6j+uw7uKbNre6gN7kmgDovd0lOF9/A5bJIoV3EiNDQF
STellfpG6CWKIiBgAeCUQ5+7bJKNpH8QcMfG35iqdjv1J/wHcT3CKZWAyxa5z52a
eF8Opw7HTPXX2JQaLl1nYMIG/1DsH4D8isEHrsFWVwoVq3Bfcz+jy2z3BiVLDBQx
Mka3mq5Y0PsFu3wQcDNwDv1a1GwRkKKnqG6pOnsCAwEAAaNLMEkwCwYDVR0PBAQD
AgQwMBMGA1UdJQQMMAoGCCsGAQUFBwMBMCUGA1UdEQQeMByCA211dIIJbXV0Lmxv
Y2FshwSsEEAOhwQKyAABMA0GCSqGSIb3DQEBCwUAA4ICAQCuoOCFR2nIcu7ml4BC
D03gA1ZJwKQ0UMg7tkm8L4VzNe1pBAefXPjHR8rRrRwVB4bdgl27QE1BWOFm46zy
2lbaYxvFmJVwWCoBAMT1MwieWKsjimDY+rrOrJEueRX/HkEiZPuldnwpo/UpzdCx
qxV7rp6aBLgUG16RFoNo0d4PoMWysORcZXB5+93M2gvLF26xuUHWjTC/7SxWwTzC
iklwZD7zsjxpRfiv8iE/pvrHCPQt1tF+coUgpy8UFa+QinWvtJsk5iMWOT523jRH
XL2HWArtXWbK0TwF5F5FXFLTne25aYbmkr9mXz/P/t4/1SMZLXlGULKNYeKqXYVK
l+ZvbAfywRbkxTKezolvEEjYVp8g9CPQgY88JpfIm9U7H4lP3WBuHbPY0dHABWrq
sv9r6G+CWaxCqP0LgQ5zMZuyAxjo0b4tiCePCLuM5jxQSOqUjbPAUkJ+6LUEmhtu
8Mo1pHTjqOBsxr4W4t2GneJGQbM4qaCCoWwMwb9x402gVbU1+1hDoxS7uoG0M9Aw
BbIEiPje6AzijTnaqS7MafQ1cundj0cC6b232ZvB/F0E9kYb59HUjiejjHcpYrA6
ZgpwvOHKyaefibeV/+6mXKfxc7e+EuKfjqB9HcXXFZgSIn6HGSJoORzLz2jWDmDf
pkCSsV+dw/7dcoUJbLRLyzjp4g==
-----END CERTIFICATE-----
```

- For **Linux**:
  - Copy the certificate data to a new file at `/etc/docker/certs.d/mut:5000/ca.crt`
  - Restart the Docker daemon
- For **Windows**:
  - Follow [this guide](https://docs.docker.com/docker-for-windows/#add-tls-certificates) to add our custom certificate.
  - Restart the Docker daemon.
- For **Mac OS**
  - Follow [this guide](https://docs.docker.com/docker-for-mac/#add-tls-certificates) to add our custom certificate.
  - Restart the Docker daemon.

> Further info: [Docker docs](https://docs.docker.com/registry/insecure/#use-self-signed-certificates)
