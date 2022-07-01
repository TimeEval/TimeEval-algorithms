FROM registry.gitlab.hpi.de/akita/i/python3-torch:0.2.1

LABEL maintainer="siddeshkanth.logonathan@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt;

COPY src /app/src
COPY algorithm.py /app/
