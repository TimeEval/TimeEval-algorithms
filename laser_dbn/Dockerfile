FROM mut:5000/akita/python3-base:0.2.1

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
COPY algorithm.py /app/
COPY laser_dbn /app/laser_dbn

RUN pip install -r /app/requirements.txt
