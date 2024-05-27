FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="ben-noah.engelhaupt@student.hpi.de,leo.wendt@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY setup.py .
COPY multinormats ./multinormats
RUN python setup.py install

COPY algorithm.py .
COPY manifest.json .
