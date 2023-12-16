FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="yannik.schroeder@student.hpi.de"
LABEL org.opencontainers.image.licenses="GPL-3.0"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app/
