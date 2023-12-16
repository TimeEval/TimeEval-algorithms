FROM ghcr.io/timeeval/python3-torch:0.3.0 as build

COPY sr /app/sr

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libfftw3-dev build-essential; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install Cython
RUN cd /app/sr/; python setup.py install

FROM ghcr.io/timeeval/python3-torch:0.3.0 as eval

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY --from=build /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY mtad_gat /app/mtad_gat
COPY manifest.json /app/
COPY algorithm.py /app/
