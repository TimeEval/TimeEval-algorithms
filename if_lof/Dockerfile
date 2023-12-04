FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="rohan.sawahn@student.hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r requirements.txt

COPY algorithm.py /app/
COPY manifest.json /app/
