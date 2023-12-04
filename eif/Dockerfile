# Separate build image!
#----------------------------
FROM ghcr.io/timeeval/python3-base:0.3.0 as build

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential;

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# build and install dependencies
RUN pip install cython==0.29.36
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
#----------------------------

# inherit from the python3-base base image
FROM ghcr.io/timeeval/python3-base:0.3.0 as runtime

LABEL maintainer="rohan.sawahn@student.hpi.de"
LABEL org.opencontainers.image.licenses=MIT

# specify the main entrypoint to your algorithm code
ENV ALGORITHM_MAIN="/app/algorithm.py"

ENV PATH="/opt/venv/bin:$PATH"

COPY --from=build /opt/venv /opt/venv

# add the algorithm code (in this case only a single file)
COPY algorithm.py /app/
COPY manifest.json /app/
