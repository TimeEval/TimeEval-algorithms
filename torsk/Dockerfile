FROM mut:5000/akita/python3-base:0.2.1

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY torsk ./torsk
COPY setup.py ./setup.py
RUN pip install .

COPY algorithm.py .
