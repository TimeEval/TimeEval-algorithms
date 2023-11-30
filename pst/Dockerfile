FROM ghcr.io/timeeval/r4-base:0.3.1

LABEL maintainer="richard.schulz@student.hpi.de"
LABEL org.opencontainers.image.licenses=MIT

# install R-install requirements
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake libnlopt0 libnlopt-dev; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install PST, TraMineR and other libraries and cleanup afterwards
RUN set -eux; \
    R -e 'install_version("TraMineR", version="2.2"); library("TraMineR")'; \
    R -e 'install_version("PST", version="0.94"); library("PST")'; \
    R -e 'install_version("arules", version="1.7"); library("arules")'; \
    R -e 'install_version("BBmisc", version="1.13"); library("BBmisc")';\
    find /opt/R/${R_VERSION}/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*

ENV ALGORITHM_MAIN="/app/algorithm.r"

COPY . /app/
