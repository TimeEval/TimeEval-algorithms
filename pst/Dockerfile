FROM registry.gitlab.hpi.de/akita/i/r4-base:0.2.1

LABEL maintainer="richard.schulz@student.hpi.de"

# install R-install requirements
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends cmake libnlopt0 libnlopt-dev; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# install PST, TraMineR and other libraries and cleanup afterwards
RUN set -eux; \
    R -e 'install.packages("TraMineR"); library("TraMineR")'; \
    R -e 'install.packages("PST"); library("PST")'; \
    R -e 'install.packages("arules"); library("arules")'; \
    R -e 'install.packages("BBmisc"); library("BBmisc")';\
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*

ENV ALGORITHM_MAIN="/app/algorithm.r"

COPY . /app/
