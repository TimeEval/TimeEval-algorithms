FROM mut:5000/akita/python3-base:latest

LABEL maintainer="yannik.schroeder@student.hpi.de"

ENV ALGORITHM_MAIN="/app/algorithm.py"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app/
