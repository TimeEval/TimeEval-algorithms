FROM ghcr.io/timeeval/python36-base:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY omni_anomaly /app/omni_anomaly
COPY manifest.json /app/
COPY algorithm.py /app/
