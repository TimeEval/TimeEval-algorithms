FROM mut:5000/akita/python3-base:0.2.1

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY telemanom /app/telemanom
COPY algorithm.py /app/
