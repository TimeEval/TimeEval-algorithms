FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY ./requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY ./algorithm.py /app/
COPY ./model.py /app/
COPY ./manifest.json /app/
