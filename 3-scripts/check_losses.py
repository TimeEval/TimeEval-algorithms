import json
import os
import docker
from pathlib import Path, PosixPath, WindowsPath
import pandas as pd
from typing import List, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import redirect_stdout

NAME = "name"
EARLY_STOPPING_DELTA = "early_stopping_delta"
DATASET_TARGET_PATH = "/data"
RESULTS_TARGET_PATH = "/results"
SCORES_FILE_NAME = "docker-algorithm-scores.csv"
MODEL_FILE_NAME = "../model.pkl"


class ExecutionType(Enum):
    TRAIN = "train"
    EXECUTE = "execute"


class DockerJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, ExecutionType):
            return o.name.lower()
        elif isinstance(o, (PosixPath, WindowsPath)):
            return str(o)
        return super().default(o)


@dataclass
class AlgorithmInterface:
    dataInput: Path
    dataOutput: Path
    modelInput: Path
    modelOutput: Path
    executionType: ExecutionType
    customParameters: dict = field(default_factory=dict)

    def to_json_string(self) -> str:
        dictionary = asdict(self)
        return json.dumps(dictionary, cls=DockerJSONEncoder)


def load_es_algorithms() -> List[Tuple[str, str]]:
    es_algorithms: List[Tuple[str, str]] = []
    dirs = Path("..")
    for manifest_path in dirs.glob("*/manifest.json"):
        manifest = json.load(manifest_path.open(mode="r"))
        training_parameters = pd.DataFrame(manifest.get("trainingStep", {}).get("parameters", []))
        if NAME in training_parameters.columns and training_parameters[training_parameters[NAME] == EARLY_STOPPING_DELTA].shape[0] > 0:
            title = str(manifest.get("title"))
            es_algorithms.append((title, str(manifest_path.parent)))
    return es_algorithms


def build_docker_image(path: str) -> str:
    image_name = f"mut:5000/akita/{path}"
    client = docker.from_env()
    print()
    print(f"Building {image_name}")
    print("----------")
    for out in client.api.build(path, rm=True, tag=image_name):
        print(json.loads(out).get("stream", ""), end='')
    return image_name


def run_docker_training(image_name: str, dataset_path: Path):
    client = docker.from_env()

    algorithm_interface = AlgorithmInterface(
        dataInput=(Path(DATASET_TARGET_PATH) / dataset_path.name).absolute(),
        dataOutput=(Path(RESULTS_TARGET_PATH) / SCORES_FILE_NAME).absolute(),
        modelInput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
        modelOutput=(Path(RESULTS_TARGET_PATH) / MODEL_FILE_NAME).absolute(),
        executionType=ExecutionType.TRAIN,
        customParameters={"epochs": 30, "early_stopping_patience": 10, "early_stopping_delta": 0.001, "split": 0.8},
    )

    container = client.containers.run(
        image=image_name,
        command=f"execute-algorithm '{algorithm_interface.to_json_string()}'",
        volumes={
            str(dataset_path.parent.absolute()): {'bind': DATASET_TARGET_PATH, 'mode': 'ro'},
            str(Path("../results").absolute()): {'bind': RESULTS_TARGET_PATH, 'mode': 'rw'}
        }
    )

    print(container)


def update_manifests():
    dirs = Path("..")
    for manifest_path in dirs.glob("*/manifest.json"):
        manifest = json.load(manifest_path.open(mode="r"))
        uses_early_stopping = False
        for parameter in manifest.get("trainingStep", {}).get("parameters", []):
            if parameter["name"] == "early_stopping_delta":
                parameter["defaultValue"] = 0.05
                parameter["description"] = "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop"
                uses_early_stopping = True
            elif parameter["name"] == "early_stopping_patience":
                parameter["defaultValue"] = 10
                parameter["description"] = "If 1 - (loss / last_loss) is less than `delta` for `patience` epochs, stop"
        if uses_early_stopping:
            json.dump(manifest, manifest_path.open(mode="w"), indent=2)



def main():
    es_algorithms = sorted(load_es_algorithms(), key=lambda x: x[0])
    print(es_algorithms)
    return

    #for algo in es_algorithms:
    #    print(algo)
    #exit()

    image_names = [build_docker_image(path) for name, path in es_algorithms]
    dataset_path = Path("../data/dataset.csv")
    with open("es-log-out.txt", "w") as log_f:
        with redirect_stdout(log_f):
            for image_name in image_names:
                print()
                print(f"Training {image_name}")
                try:
                    run_docker_training(image_name, dataset_path)
                except Exception as e:
                    print(e)
                print("----------------------")
                print()


if __name__ == "__main__":
    update_manifests()
