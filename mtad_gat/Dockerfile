FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.5 as build

COPY sr /app/sr

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libfftw3-dev build-essential; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install Cython
RUN cd /app/sr/; python setup.py install

FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.5 as eval

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY --from=build /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY mtad_gat /app/mtad_gat
COPY algorithm.py /app/
