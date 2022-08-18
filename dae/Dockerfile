# inherit from the python3-base base image
FROM registry.gitlab.hpi.de/akita/i/python3-base:0.2.5

# specify the main entrypoint to your algorithm code
ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# add the algorithm code (in this case only a single file)
COPY algorithm.py /app/

COPY model.py /app/
