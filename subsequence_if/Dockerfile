FROM ghcr.io/timeeval/pyod:0.3.1

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY manifest.json /app/
COPY algorithm.py /app/
