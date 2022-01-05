FROM mut:5000/akita/python3-torch:latest

LABEL maintainer="phillip.wenig@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY mscred /app/mscred
COPY algorithm.py /app/
