FROM ghcr.io/timeeval/python3-torch:0.3.0

LABEL maintainer="siddeshkanth.logonathan@student.hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt;

COPY src /app/src
COPY manifest.json /app/
COPY algorithm.py /app/
