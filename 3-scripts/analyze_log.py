import re
import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("es-log-out.txt", "r") as f:
        logs = f.read()

    algos = logs.split("Training registry.gitlab.hpi.de/akita/i/")
    del algos[0]

    loss_progression = dict()
    for algo in algos:
        title = algo.split("\n")[0]
        if "Early Stopping: " in algo:
            losses = list(map(lambda x: float(x.split("Early Stopping: ")[1]), re.findall(r"Early Stopping: \d*[.]\d*", algo)))
        else:
            losses = list(map(lambda x: float(x.split("val_loss: ")[1]), re.findall(r"val_loss: \d*[.]\d*", algo)))

        loss_progression[title] = losses

    for k, v in loss_progression.items():
        print(f"{k}: {len(v)}")
        v = np.array(v)
        deltas = np.abs(v - np.roll(v, -1))
        if v.shape[0] > 0:
            plt.plot(deltas / np.roll(v, -1), alpha=0.5, label=k)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
