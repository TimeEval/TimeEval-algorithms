FROM openjdk:11-jre-slim-buster

LABEL maintainer="sebastian.schmidl@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

VOLUME ["/data", "/results"]

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "execute-algorithm" ]
