FROM mut:5000/akita/r4-base:0.2.1

LABEL maintainer="richard.schulz@student.hpi.de"

# install stream and other libraries and cleanup afterwards
RUN set -eux; \
    R -e 'install.packages("BBmisc")';\
    R -e 'install.packages("stream")';\
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*

ENV ALGORITHM_MAIN="/app/algorithm.r"

COPY . /app/
