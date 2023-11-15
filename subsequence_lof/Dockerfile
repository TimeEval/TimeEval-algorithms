FROM ghcr.io/hpi-information-systems/pyod:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.source=https://github.com/HPI-Information-Systems/TimeEval-algorithms
LABEL org.opencontainers.image.licenses=MIT
# LABEL org.opencontainers.image.description=""

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY algorithm.py .
