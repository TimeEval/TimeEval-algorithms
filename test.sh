#!/usr/bin/env bash

# call as: ./test.sh <algorithm_name> [test] [execute]

algorithm=$1 # <algorithm_name>

docker rmi registry.gitlab.hpi.de/akita/i/$algorithm:latest

docker build -t registry.gitlab.hpi.de/akita/i/$algorithm $algorithm

docker rm $(docker ps -a -q)

for execution in "$@"
do
  if [[ $execution == "train" || $execution == "execute" ]]
  then
    docker run --rm -v $(pwd)/1-data:/data:ro -v $(pwd)/2-results:/results:rw registry.gitlab.hpi.de/akita/i/$algorithm:latest execute-algorithm '{"executionType": "'$execution'", "dataInput": "/data/dataset.csv", "dataOutput": "/results/anomaly_scores.ts", "modelInput": "/model/model.txt", "modelOutput": "/model/model.txt"}'
  fi
done
