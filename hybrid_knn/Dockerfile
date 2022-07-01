FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.1

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY hybrid_knn /app/hybrid_knn
COPY algorithm.py /app/algorithm.py
COPY manifest.json /app/manifest.json
