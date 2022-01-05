FROM mut:5000/akita/python3-torch:0.2.1

LABEL maintainer="yannik.schroeder@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY . /app
