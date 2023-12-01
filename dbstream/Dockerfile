FROM ghcr.io/timeeval/r4-base:0.3.1

LABEL maintainer="richard.schulz@student.hpi.de"
LABEL org.opencontainers.image.licenses=MIT

# install stream and other libraries and cleanup afterwards
RUN set -eux; \
    R -e 'install_version("BBmisc", version="1.12"); library("BBmisc")';\
    R -e 'install_version("stream", version="1.5.1"); library("stream")';\
    find /opt/R/${R_VERSION}/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*

ENV ALGORITHM_MAIN="/app/algorithm.r"

COPY . /app/
