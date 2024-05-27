FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5 as base

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install C dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libfftw3-3; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Separate build image!
#----------------------------
FROM base as build

# install build dependencies
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends libfftw3-dev build-essential; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# build norm and its dependencies
COPY C ./C
COPY setup.py .
COPY normats ./normats
RUN python setup.py install
#----------------------------


FROM base

COPY --from=build /opt/venv /opt/venv
COPY algorithm.py .

ENV PATH="/opt/venv/bin:$PATH"
