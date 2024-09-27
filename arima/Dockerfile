FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="thorsten.papenbrock@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt;

COPY ptsa /app/ptsa
COPY algorithm.py /app/
