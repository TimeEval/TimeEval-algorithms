FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.1

LABEL maintainer="yannik.schroeder@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY tsbitmapper.py /app/
COPY algorithm.py /app/
