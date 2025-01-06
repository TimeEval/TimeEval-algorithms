FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="yannik.schroeder@tu-berlin.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY manifest.json /app/
COPY hst.py /app/
COPY algorithm.py /app/
