FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="thorsten.papenbrock@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt;

COPY ptsa /app/ptsa
COPY algorithm.py /app/
