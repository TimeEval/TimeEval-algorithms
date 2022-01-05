FROM debian:buster-slim

LABEL maintainer="sebastian.schmidl@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends less wget locales ca-certificates fonts-texgyre; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure default locale
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8

# Install R
ENV R_BASE_VERSION 3.5.2-1

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential libopenblas-base r-base=${R_BASE_VERSION}; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/* /usr/share/R/doc/*; \
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    find /usr/lib/R/library/translations/* -type d \
      \( \
          -name en -prune \
          -o -name de -prune \
          -o -name en@quot -prune \
      \) -o -exec rm -rf '{}' +;

# Set default CRAN mirror
COPY Rprofile.site /etc/R/Rprofile.site

# Install common dependencies
RUN set -eux; \
    R -e 'install.packages("jsonlite")'; \
    find /usr/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/*

# Setup TimeEval integration

VOLUME ["/data", "/results"]

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "execute-algorithm" ]
