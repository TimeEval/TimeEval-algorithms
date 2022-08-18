FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
COPY algorithm.py /app/
COPY laser_dbn /app/laser_dbn

RUN pip install -r /app/requirements.txt
