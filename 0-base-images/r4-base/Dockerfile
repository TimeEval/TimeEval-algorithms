FROM debian:buster-slim

LABEL maintainer="phillip.wenig@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y --no-install-recommends less wget locales ca-certificates fonts-texgyre gnupg; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Configure default locale
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
	&& locale-gen en_US.utf8 \
	&& /usr/sbin/update-locale LANG=en_US.UTF-8

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV R_VERSION=4.2.0

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential libopenblas-base; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install R
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gdebi; \
    wget https://cdn.rstudio.com/r/debian-10/pkgs/r-${R_VERSION}_1_amd64.deb; \
    gdebi -n r-${R_VERSION}_1_amd64.deb; \
    ln -s /opt/R/${R_VERSION}/bin/R /usr/local/bin/R; \
    ln -s /opt/R/${R_VERSION}/bin/Rscript /usr/local/bin/Rscript; \
    /opt/R/${R_VERSION}/bin/R --version; \
    rm -f r-${R_VERSION}_1_amd64.deb; \
    apt-get purge -y gdebi && apt-get autoremove -y --purge; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /usr/share/doc/* /usr/share/R/doc/*; \
    find /opt/R/${R_VERSION}/lib/R/library -depth \
      \( \
        -type d -a \( -name doc -o -name html \) \
      \) -exec rm -rf '{}' +; \
    find /opt/R/${R_VERSION}/lib/R/library/translations/* -type d \
      \( \
          -name en -prune \
          -o -name de -prune \
          -o -name en@quot -prune \
      \) -o -exec rm -rf '{}' +;

# Set default CRAN mirror
COPY Rprofile.site /opt/R/${R_VERSION}/lib/R/etc/Rprofile.site

# Install common dependencies
RUN set -eux; \
    R -e 'install.packages("jsonlite")'; \
    find /opt/R/${R_VERSION}/lib/R/library -depth \
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
