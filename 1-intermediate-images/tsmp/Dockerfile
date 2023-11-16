FROM registry.gitlab.hpi.de/akita/i/r-base

LABEL maintainer="sebastian.schmidl@hpi.de"

# install tsmp library and cleanup afterwards
RUN set -eux; \
    R -e 'install.packages("tsmp")'; \
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*
