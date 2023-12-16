FROM ghcr.io/timeeval/python3-base:0.3.0


LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses="GPL-3.0"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install python dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY model.py /app/
COPY manifest.json /app/
COPY algorithm.py /app/
