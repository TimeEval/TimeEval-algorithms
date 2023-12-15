FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY multi_hmm /app/multi_hmm
COPY manifest.json /app/
COPY algorithm.py /app/
