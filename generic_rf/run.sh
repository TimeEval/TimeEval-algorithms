#!/usr/bin/env bash

custom_params=${1:-'{}'}
echo ""
echo "=== Training ==="
python algorithm.py "{ \"executionType\": \"train\", \"dataInput\": \"../data/dataset.csv\", \"dataOutput\": \"./scores.csv\", \"modelInput\": \"./model.pkl\", \"modelOutput\": \"./model.pkl\", \"customParameters\": $custom_params }"


echo ""
echo "=== Execution ==="
python algorithm.py "{ \"executionType\": \"execute\", \"dataInput\": \"../data/dataset.csv\", \"dataOutput\": \"./scores.csv\", \"modelInput\": \"./model.pkl\", \"modelOutput\": \"./model.pkl\" }"
