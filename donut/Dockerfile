FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY donut /app/donut
COPY manifest.json /app/
COPY algorithm.py /app/
