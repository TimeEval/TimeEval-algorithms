FROM mut:5000/akita/python3-base:0.2.1

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./algorithm.py .
COPY ./model.py .
COPY ./manifest.json .
