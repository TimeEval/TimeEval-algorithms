FROM ghcr.io/timeeval/r4-base:0.3.1

LABEL maintainer="sebastian.schmidl@hpi.de"

# install tsmp library and cleanup afterwards
RUN set -eux; \
    R -e 'install_version("tsmp", version="0.4.15"); library("tsmp")'; \
    find /opt/R/${R_VERSION}/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*
