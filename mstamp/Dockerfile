FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY manifest.json /app/
COPY algorithm.py /app/
