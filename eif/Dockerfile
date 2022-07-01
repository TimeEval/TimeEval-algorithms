# Separate build image!
#----------------------------
FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1 as build

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential;

RUN python -m venv --system-site-packages /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# build and install dependencies
RUN pip install cython
COPY requirements.txt .
RUN pip install -r ./requirements.txt
#----------------------------

# inherit from the python3-base base image
FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1 as runtime

LABEL maintainer="rohan.sawahn@student.hpi.de"

# specify the main entrypoint to your algorithm code
ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY --from=build /opt/venv /opt/venv

# add the algorithm code (in this case only a single file)
COPY algorithm.py .

ENV PATH="/opt/venv/bin:$PATH"
