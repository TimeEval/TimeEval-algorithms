FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
RUN pip install --no-index torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html

COPY algorithm.py /app/
COPY GDN /app/GDN
