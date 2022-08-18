FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY algorithm.py /app/
COPY multi_hmm /app/multi_hmm
