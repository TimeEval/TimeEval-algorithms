FROM python:3.6-slim-buster

LABEL maintainer="phillip.wenig@hpi.de"

RUN set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

VOLUME ["/data", "/results"]

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh

COPY requirements.txt /tmp/
# install requirements and cleanup afterwards (also removes tests and cached cython files of the dependencies)
RUN set -eux; \
    pip install --no-cache-dir -r /tmp/requirements.txt; \
    find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip


ENTRYPOINT [ "/entrypoint.sh" ]
CMD [ "execute-algorithm" ]
