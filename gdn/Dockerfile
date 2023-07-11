FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.6

LABEL maintainer="2er0@dbaumi.at"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN apt-get update; \
    apt-get install -y gcc python3-dev; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN pip install -r /app/requirements.txt

COPY algorithm.py /app/
COPY GDN /app/GDN
