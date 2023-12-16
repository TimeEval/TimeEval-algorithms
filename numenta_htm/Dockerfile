FROM ghcr.io/timeeval/python2-base:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses="GPL-3.0"

ENV ALGORITHM_MAIN="/app/algorithm.py"

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install nupic

# overwrite certain library versions
RUN pip install python-dateutil==2.5.0

COPY ./htm /app/htm
COPY manifest.json /app/
COPY algorithm.py /app/
