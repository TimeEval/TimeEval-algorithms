FROM ghcr.io/timeeval/python3-torch:0.3.0

LABEL maintainer="phillip.wenig@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY ocean_wnn /app/ocean_wnn
COPY manifest.json /app/
COPY algorithm.py /app/
