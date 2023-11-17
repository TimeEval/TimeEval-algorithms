FROM ghcr.io/timeeval/r-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

# install tsmp library and cleanup afterwards
RUN set -eux; \
    R -e 'install.packages("tsmp")'; \
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*
