# inherit from the python3-base base image
FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="sebastian.schmidl@hpi.de"
LABEL org.opencontainers.image.licenses=MIT

# specify the main entrypoint to your algorithm code
ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# add the algorithm code
COPY manifest.json /app/
COPY model.py /app/
COPY algorithm.py /app/
