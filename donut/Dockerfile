FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY donut /app/donut
COPY algorithm.py /app/
