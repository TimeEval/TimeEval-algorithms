import numpy as np
import pandas as pd
import argparse
import json
import sys

from nupic.frameworks.opf.model_factory import ModelFactory
from htm.model_params import get_model_parameters


class AlgorithmArgs(argparse.Namespace):
    @staticmethod
    def get_default_custom_parameters():
        return {
            "encoding_input_width": 21,
            "encoding_output_width": 50,
            "autoDetectWaitRecords": 50,
            "columnCount": 2048,
            "numActiveColumnsPerInhArea": 40,
            "potentialPct": 0.5,
            'synPermConnected': 0.1,
            'synPermActiveInc': 0.1,
            'synPermInactiveDec': 0.005,
            'cellsPerColumn': 32,
            'inputWidth': 2048,
            'newSynapseCount': 20,
            'maxSynapsesPerSegment': 32,
            'maxSegmentsPerCell': 128,
            'initialPerm': 0.21,
            'permanenceInc': 0.1,
            'permanenceDec' : 0.1,
            'globalDecay': 0.0,
            'maxAge': 0,
            'minThreshold': 9,
            'activationThreshold': 12,
            'pamLength': 1,
            'alpha': 0.5,
            'random_state': 42
        }

    @property
    def ts(self):
        dataset = self.df
        if type(dataset.timestamp[0]) == str:
            dataset["timestamp"] = pd.to_datetime(dataset.timestamp.astype(int))
        return dataset.values[:, 0:2]

    @property
    def df(self):
        return pd.read_csv(self.dataInput, parse_dates=["timestamp"], infer_datetime_format=True)

    @staticmethod
    def from_sys_args():
        args = json.loads(sys.argv[1])
        customParameters = AlgorithmArgs.get_default_custom_parameters()
        customParameters.update(args.get("customParameters", {}))
        args["customParameters"] = customParameters
        return AlgorithmArgs(**args)


def _build_model(data, args):
    params = get_model_parameters()
    params["modelParams"]["sensorParams"]["encoders"]["value"]["minval"] = data[:, 1].min()
    params["modelParams"]["sensorParams"]["encoders"]["value"]["maxval"] = data[:, 1].max()
    params["modelParams"]["sensorParams"]["encoders"]["value"]["w"] = args.customParameters["encoding_input_width"]
    params["modelParams"]["sensorParams"]["encoders"]["value"]["n"] = args.customParameters["encoding_output_width"]
    params["modelParams"]["anomalyParams"]["autoDetectWaitRecords"] = args.customParameters["autoDetectWaitRecords"]

    params["modelParams"]["spParams"]["columnCount"] = args.customParameters["columnCount"]
    params["modelParams"]["spParams"]["numActiveColumnsPerInhArea"] = args.customParameters["numActiveColumnsPerInhArea"]
    params["modelParams"]["spParams"]["potentialPct"] = args.customParameters["potentialPct"]
    params["modelParams"]["spParams"]["synPermConnected"] = args.customParameters["synPermConnected"]
    params["modelParams"]["spParams"]["synPermActiveInc"] = args.customParameters["synPermActiveInc"]
    params["modelParams"]["spParams"]["synPermInactiveDec"] = args.customParameters["synPermInactiveDec"]
    params["modelParams"]["spParams"]["seed"] = args.customParameters["random_state"]

    params["modelParams"]["tmParams"]["columnCount"] = args.customParameters["columnCount"]
    params["modelParams"]["tmParams"]["cellsPerColumn"] = args.customParameters["cellsPerColumn"]
    params["modelParams"]["tmParams"]["inputWidth"] = args.customParameters["inputWidth"]
    params["modelParams"]["tmParams"]["newSynapseCount"] = args.customParameters["newSynapseCount"]
    params["modelParams"]["tmParams"]["maxSynapsesPerSegment"] = args.customParameters["maxSynapsesPerSegment"]
    params["modelParams"]["tmParams"]["maxSegmentsPerCell"] = args.customParameters["maxSegmentsPerCell"]
    params["modelParams"]["tmParams"]["initialPerm"] = args.customParameters["initialPerm"]
    params["modelParams"]["tmParams"]["permanenceInc"] = args.customParameters["permanenceInc"]
    params["modelParams"]["tmParams"]["permanenceDec"] = args.customParameters["permanenceDec"]
    params["modelParams"]["tmParams"]["globalDecay"] = args.customParameters["globalDecay"]
    params["modelParams"]["tmParams"]["maxAge"] = args.customParameters["maxAge"]
    params["modelParams"]["tmParams"]["minThreshold"] = args.customParameters["minThreshold"]
    params["modelParams"]["tmParams"]["activationThreshold"] = args.customParameters["activationThreshold"]
    params["modelParams"]["tmParams"]["pamLength"] = args.customParameters["pamLength"]
    params["modelParams"]["tmParams"]["seed"] = args.customParameters["random_state"]

    params["modelParams"]["clParams"]["alpha"] = args.customParameters["alpha"]


    model = ModelFactory.create(params)
    model.enableInference({'predictedField': 'value'})

    return model


def execute(args):
    data = args.ts
    model = _build_model(data, args)

    scores = []
    for row in data:
        result = model.run({
            "timestamp": row[0],
            "value": row[1]
        })
        scores.append(result.inferences["anomalyScore"])

    scores = np.array(scores)
    scores.tofile(args.dataOutput, sep="\n")


def set_random_state(config):
    seed = config.customParameters["random_state"]
    import random
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = AlgorithmArgs.from_sys_args()
    set_random_state(args)

    if args.executionType == "train":
        print "No training required!"
    elif args.executionType == "execute":
        execute(args)
    else:
        raise ValueError(format("No executionType '%s' available! Choose either 'train' or 'execute'.", args.executionType))
