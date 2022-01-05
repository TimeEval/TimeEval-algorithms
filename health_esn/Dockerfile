FROM mut:5000/akita/python3-base:0.2.1

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY health_esn ./health_esn
COPY algorithm.py .
