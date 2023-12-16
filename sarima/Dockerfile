FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
# custom license has no abbreviation

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN set -eux; \
    pip install --no-cache-dir -r /app/requirements.txt; \
    find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip

COPY sarima.py /app/
COPY LICENSE /app/
COPY manifest.json /app/
COPY algorithm.py /app/
