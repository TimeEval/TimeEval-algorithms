FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

LABEL maintainer="sebastian.schmidl@hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY torsk ./torsk
COPY setup.py ./setup.py
RUN pip install .

COPY algorithm.py .
