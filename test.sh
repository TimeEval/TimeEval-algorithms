#!/usr/bin/env bash

# call as: ./test.sh <algorithm_name> [test] [execute]

algorithm=$1 # <algorithm_name>

docker rmi mut:5000/akita/$algorithm:latest

docker build -t mut:5000/akita/$algorithm $algorithm

docker rm $(docker ps -a -q)

for execution in "$@"
do
  if [[ $execution == "train" || $execution == "execute" ]]
  then
    docker run --rm -v $(pwd)/1-data:/data:ro -v $(pwd)/2-results:/results:rw mut:5000/akita/$algorithm:latest execute-algorithm '{"executionType": "'$execution'", "dataInput": "/data/dataset.csv", "dataOutput": "/results/anomaly_scores.ts", "modelInput": "/model/model.txt", "modelOutput": "/model/model.txt"}'
  fi
done
