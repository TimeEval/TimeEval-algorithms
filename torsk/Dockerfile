FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY torsk /app/torsk
COPY setup.py /app/setup.py
RUN pip install /app

COPY manifest.json /app/
COPY algorithm.py /app/
