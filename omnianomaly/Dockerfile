FROM registry.gitlab.hpi.de/akita/i/python36-base:0.2.5

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY omni_anomaly /app/omni_anomaly
COPY algorithm.py /app/
