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

# build and install matrix profile lib
COPY matrix_profile /matrix_profile
RUN cd /matrix_profile && python setup.py install
#----------------------------


FROM base

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=build /opt/venv /opt/venv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY SAND.py .
COPY algorithm.py .
