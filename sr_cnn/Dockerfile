FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.1 as build

COPY anomalydetector /app/anomalydetector

RUN apt-get update
RUN apt-get install build-essential -y
RUN pip install --no-cache-dir Cython tqdm torchvision==0.8.2
RUN cd /app/anomalydetector/; python setup.py install

FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.1 as eval

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY --from=build /usr/local/lib/python3.7/site-packages /usr/local/lib/python3.7/site-packages
COPY srcnn /app/srcnn
COPY algorithm.py /app/
