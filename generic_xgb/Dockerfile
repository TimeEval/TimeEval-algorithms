FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./algorithm.py .
COPY ./model.py .
COPY ./manifest.json .
