FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"

# install pyod library and cleanup afterwards
RUN set -eux; \
    pip install --no-cache-dir pyod==1.1.2; \
    find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip
