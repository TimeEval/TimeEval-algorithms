FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.5

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY tanogan /app/tanogan
COPY algorithm.py /app/
