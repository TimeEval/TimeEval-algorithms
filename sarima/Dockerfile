FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt .
RUN set -eux; \
    pip install --no-cache-dir -r requirements.txt; \
    find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' +; \
    rm -rf /tmp/* /var/tmp/* ~/.cache/pip

COPY sarima.py .
COPY algorithm.py .
